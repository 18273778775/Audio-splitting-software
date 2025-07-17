#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频分割工具
支持MP3和WAV格式的音频文件分割
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import math

try:
    import librosa
    import soundfile as sf
    import numpy as np
    from scipy import signal
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from video_processor import VideoProcessor
except ImportError as e:
    print(f"请安装必要的处理库:")
    print(f"pip install librosa soundfile numpy scipy matplotlib opencv-python")
    print(f"错误详情: {e}")
    sys.exit(1)


class AudioSplitter:
    """音频分割核心类"""
    
    def __init__(self):
        self.supported_formats = ['.mp3', '.wav']
        self.video_processor = VideoProcessor()
    
    def is_supported_format(self, file_path):
        """检查文件格式是否支持"""
        return Path(file_path).suffix.lower() in self.supported_formats

    def analyze_audio_volume(self, audio_data, sample_rate, window_size=0.02):
        """
        分析音频的音量变化

        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            window_size: 分析窗口大小（秒），默认0.02秒提高精度

        Returns:
            tuple: (时间轴, RMS音量数组)
        """
        # 计算窗口大小（采样点数）
        window_samples = int(window_size * sample_rate)

        # 计算RMS音量，使用更小的步长提高精度
        rms_values = []
        time_points = []

        # 使用更小的步长：窗口大小的1/4，提高时间精度
        step_samples = max(1, window_samples // 4)

        for i in range(0, len(audio_data) - window_samples, step_samples):
            window_data = audio_data[i:i + window_samples]
            rms = np.sqrt(np.mean(window_data ** 2))
            rms_values.append(rms)
            # 使用更精确的时间计算
            time_points.append(i / sample_rate)

        return np.array(time_points), np.array(rms_values)

    def find_silence_regions(self, audio_data, sample_rate, silence_threshold=0.01, min_silence_duration=0.1):
        """
        检测音频中的静音区域

        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            silence_threshold: 静音阈值（RMS值）
            min_silence_duration: 最小静音持续时间（秒）

        Returns:
            list: 静音区域列表 [(start_time, end_time), ...]
        """
        time_points, rms_values = self.analyze_audio_volume(audio_data, sample_rate)

        # 找到低于阈值的区域
        silence_mask = rms_values < silence_threshold

        # 找到连续的静音区域
        silence_regions = []
        in_silence = False
        silence_start = 0

        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                # 静音开始
                in_silence = True
                silence_start = time_points[i]
            elif not is_silent and in_silence:
                # 静音结束
                in_silence = False
                silence_end = time_points[i]

                # 检查静音持续时间
                if silence_end - silence_start >= min_silence_duration:
                    silence_regions.append((silence_start, silence_end))

        # 处理音频结尾的静音
        if in_silence:
            silence_end = time_points[-1] if len(time_points) > 0 else 0
            if silence_end - silence_start >= min_silence_duration:
                silence_regions.append((silence_start, silence_end))

        return silence_regions

    def find_optimal_split_point(self, audio_data, sample_rate, target_time, search_range=2.0):
        """
        在目标时间点附近寻找最佳分割点

        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            target_time: 目标分割时间（秒）
            search_range: 搜索范围（秒），在目标时间前后此范围内搜索

        Returns:
            float: 最佳分割时间点
        """
        # 计算搜索范围
        start_time = max(0, target_time - search_range / 2)
        end_time = min(len(audio_data) / sample_rate, target_time + search_range / 2)

        # 获取搜索范围内的音频数据
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        search_audio = audio_data[start_sample:end_sample]

        if len(search_audio) == 0:
            return target_time

        # 分析这段音频的音量，使用更高精度
        time_points, rms_values = self.analyze_audio_volume(search_audio, sample_rate, window_size=0.01)

        if len(rms_values) == 0:
            return target_time

        # 找到音量最低的点
        min_volume_idx = np.argmin(rms_values)
        optimal_time = start_time + time_points[min_volume_idx]

        # 确保精度：四舍五入到最近的采样点
        optimal_sample = round(optimal_time * sample_rate)
        optimal_time = optimal_sample / sample_rate

        return optimal_time
    
    def split_audio(self, file_path, segment_duration=None, custom_durations=None,
                   smart_split=False, search_range=2.0, progress_callback=None):
        """
        分割音频文件

        Args:
            file_path (str): 音频文件路径
            segment_duration (int, optional): 固定分割时长（秒）
            custom_durations (list, optional): 自定义长度数组（秒）
            smart_split (bool): 是否启用智能分割
            search_range (float): 智能分割搜索范围（秒）
            progress_callback (callable): 进度回调函数

        Returns:
            tuple: (success, message, output_files)
        """
        try:
            # 参数验证
            if segment_duration is None and custom_durations is None:
                return False, "必须指定分割时长或自定义长度数组", []

            if segment_duration is not None and custom_durations is not None:
                return False, "不能同时指定固定时长和自定义长度", []

            # 检查文件是否存在
            if not os.path.exists(file_path):
                return False, "文件不存在", []

            # 检查文件格式
            if not self.is_supported_format(file_path):
                return False, "不支持的文件格式，仅支持MP3和WAV格式", []

            # 加载音频文件
            if progress_callback:
                progress_callback(0, "正在加载音频文件...")

            # 使用librosa加载音频文件
            audio_data, sample_rate = librosa.load(file_path, sr=None)

            # 计算分割参数
            total_duration = len(audio_data) / sample_rate  # 秒

            # 根据分割模式处理
            if custom_durations is not None:
                # 自定义长度模式
                return self._split_audio_custom(audio_data, sample_rate, total_duration,
                                              custom_durations, file_path, smart_split, search_range, progress_callback)
            else:
                # 固定时长模式（原有逻辑）
                return self._split_audio_fixed(audio_data, sample_rate, total_duration,
                                             segment_duration, file_path, smart_split, search_range, progress_callback)

        except Exception as e:
            return False, f"分割过程中发生错误: {str(e)}", []

    def split_audio_by_video_durations(self, audio_file_path, video_durations,
                                     smart_split=False, search_range=2.0, progress_callback=None):
        """
        根据视频时长列表分割音频

        Args:
            audio_file_path (str): 音频文件路径
            video_durations (list): 视频时长列表（秒）
            smart_split (bool): 是否启用智能分割
            search_range (float): 智能分割搜索范围（秒）
            progress_callback (callable): 进度回调函数

        Returns:
            tuple: (success, message, output_files)
        """
        try:
            # 验证输入
            if not audio_file_path or not os.path.exists(audio_file_path):
                return False, "音频文件不存在", []

            if not self.is_supported_format(audio_file_path):
                return False, f"不支持的音频格式: {Path(audio_file_path).suffix}", []

            if not video_durations or len(video_durations) == 0:
                return False, "视频时长列表不能为空", []

            # 检查所有时长都是正数
            for i, duration in enumerate(video_durations):
                if duration <= 0:
                    return False, f"第{i+1}个视频时长必须大于0秒", []

            # 使用自定义长度分割模式，传入视频时长作为自定义时长
            return self.split_audio(
                audio_file_path,
                custom_durations=video_durations,
                smart_split=smart_split,
                search_range=search_range,
                progress_callback=progress_callback
            )

        except Exception as e:
            return False, f"视频时长匹配分割过程中发生错误: {str(e)}", []

    def _split_audio_fixed(self, audio_data, sample_rate, total_duration, segment_duration, file_path, smart_split, search_range, progress_callback):
        """固定时长分割模式"""
        if segment_duration >= total_duration:
            return False, "分割时长大于或等于音频总时长", []

        # 计算分割数量
        num_segments = math.ceil(total_duration / segment_duration)

        # 创建输出目录
        input_path = Path(file_path)
        output_dir = input_path.parent / "output"
        output_dir.mkdir(exist_ok=True)

        # 生成输出文件名前缀
        base_name = input_path.stem
        file_extension = input_path.suffix

        output_files = []

        # 计算分割点 - 所有模式都使用高精度算法
        split_points = []
        current_position = 0
        split_points.append(current_position)  # 起始点

        for i in range(1, num_segments):
            target_time = current_position + segment_duration

            if smart_split:
                # 智能分割：寻找最佳分割点
                optimal_time = self.find_optimal_split_point(audio_data, sample_rate, target_time, search_range)
                split_points.append(optimal_time)
                current_position = optimal_time  # 更新当前位置，确保下一段从这里开始
            else:
                # 高精度固定分割：确保分割点对齐到采样点边界
                # 使用round()确保精确对齐到采样点
                target_sample = round(target_time * sample_rate)
                precise_time = target_sample / sample_rate
                split_points.append(precise_time)
                current_position = precise_time  # 更新当前位置，确保连续性

        split_points.append(total_duration)  # 添加结束点

        # 执行分割
        for i in range(num_segments):
            start_time = split_points[i]
            end_time = split_points[i + 1]

            # 使用round()提高精度，避免截断误差
            start_sample = round(start_time * sample_rate)
            end_sample = round(end_time * sample_rate)
            end_sample = min(end_sample, len(audio_data))

            # 提取音频片段
            segment_data = audio_data[start_sample:end_sample]

            # 生成输出文件名
            output_filename = f"{base_name}_part_{i+1:03d}{file_extension}"
            output_path = output_dir / output_filename

            # 保存音频片段
            sf.write(str(output_path), segment_data, sample_rate)
            output_files.append(str(output_path))

            # 更新进度
            if progress_callback:
                progress = int((i + 1) / num_segments * 100)
                split_type = "智能" if smart_split else "固定"
                progress_callback(progress, f"正在{split_type}分割第 {i+1}/{num_segments} 个片段...")

        split_type = "智能" if smart_split else "固定"
        return True, f"{split_type}分割完成！共生成 {num_segments} 个文件", output_files

    def _split_audio_custom(self, audio_data, sample_rate, total_duration, custom_durations, file_path, smart_split, search_range, progress_callback):
        """自定义长度分割模式"""
        # 验证自定义长度数组
        if not custom_durations or len(custom_durations) == 0:
            return False, "自定义长度数组不能为空", []

        # 检查所有长度都是正数
        for i, duration in enumerate(custom_durations):
            if duration <= 0:
                return False, f"第{i+1}个长度必须大于0秒", []

        # 计算指定段的总时长
        specified_total = sum(custom_durations)

        # 计算最后一段的长度
        last_segment_duration = total_duration - specified_total

        # 检查边界条件
        if specified_total >= total_duration:
            return False, f"指定的总时长({specified_total:.1f}秒)大于或等于音频总时长({total_duration:.1f}秒)", []

        # 确定是否创建最后一段
        create_last_segment = last_segment_duration >= 1.0  # 最后一段至少1秒

        # 计算总段数
        total_segments = len(custom_durations) + (1 if create_last_segment else 0)

        # 创建输出目录
        input_path = Path(file_path)
        output_dir = input_path.parent / "output"
        output_dir.mkdir(exist_ok=True)

        # 生成输出文件名前缀
        base_name = input_path.stem
        file_extension = input_path.suffix

        output_files = []

        # 计算分割点
        split_points = [0]  # 起始点
        current_position = 0

        for duration in custom_durations:
            target_time = current_position + duration
            if smart_split:
                # 智能分割：寻找最佳分割点
                optimal_time = self.find_optimal_split_point(audio_data, sample_rate, target_time, search_range)
                split_points.append(optimal_time)
                current_position = optimal_time  # 更新当前位置，确保连贯性
            else:
                # 高精度固定分割：确保分割点对齐到采样点边界
                # 使用round()确保精确对齐到采样点
                target_sample = round(target_time * sample_rate)
                precise_time = target_sample / sample_rate
                split_points.append(precise_time)
                current_position = precise_time  # 更新当前位置，确保连续性

        # 分割指定长度的段
        for i, duration in enumerate(custom_durations):
            start_time = split_points[i]
            end_time = split_points[i + 1]

            # 使用round()提高精度，避免截断误差
            start_sample = round(start_time * sample_rate)
            end_sample = round(end_time * sample_rate)

            # 确保不超出音频范围
            end_sample = min(end_sample, len(audio_data))

            # 提取音频片段
            segment_data = audio_data[start_sample:end_sample]

            # 生成输出文件名
            output_filename = f"{base_name}_part_{i+1:03d}{file_extension}"
            output_path = output_dir / output_filename

            # 保存音频片段
            sf.write(str(output_path), segment_data, sample_rate)
            output_files.append(str(output_path))

            # 更新进度
            if progress_callback:
                progress = int((i + 1) / total_segments * 100)
                split_type = "智能" if smart_split else "自定义"
                progress_callback(progress, f"正在{split_type}分割第 {i+1}/{total_segments} 个片段...")

        # 处理最后一段（如果需要）
        if create_last_segment:
            start_time = split_points[-1]
            # 使用round()提高精度
            start_sample = round(start_time * sample_rate)
            end_sample = len(audio_data)

            # 提取音频片段
            segment_data = audio_data[start_sample:end_sample]

            # 生成输出文件名
            segment_index = len(custom_durations) + 1
            output_filename = f"{base_name}_part_{segment_index:03d}{file_extension}"
            output_path = output_dir / output_filename

            # 保存音频片段
            sf.write(str(output_path), segment_data, sample_rate)
            output_files.append(str(output_path))

            # 更新进度
            if progress_callback:
                split_type = "智能" if smart_split else "自定义"
                progress_callback(100, f"正在{split_type}分割第 {total_segments}/{total_segments} 个片段...")

        # 生成结果消息
        split_type = "智能" if smart_split else "自定义"
        message = f"{split_type}分割完成！共生成 {len(output_files)} 个文件"
        if not create_last_segment:
            message += f"（最后一段时长{last_segment_duration:.1f}秒，小于1秒，已跳过）"

        return True, message, output_files


class WaveformViewer:
    """音频波形可视化窗口"""

    def __init__(self, parent, audio_file, splitter):
        self.parent = parent
        self.audio_file = audio_file
        self.splitter = splitter
        self.audio_data = None
        self.sample_rate = None
        self.split_points = []
        self.selected_point_index = None
        self.dragging = False

        # 创建新窗口
        self.window = tk.Toplevel(parent)
        self.window.title("音频波形可视化")
        self.window.geometry("1000x600")
        self.window.resizable(True, True)

        self.setup_ui()
        self.load_and_display_audio()

    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="波形控制", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 分割点信息
        self.info_label = ttk.Label(control_frame, text="加载中...")
        self.info_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        # 按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, sticky=tk.W)

        ttk.Button(button_frame, text="刷新波形", command=self.refresh_waveform).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="添加分割点", command=self.add_split_point).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="删除选中点", command=self.delete_selected_point).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="清除所有点", command=self.clear_split_points).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="关闭", command=self.window.destroy).pack(side=tk.LEFT)

        # 波形显示区域
        self.figure = plt.Figure(figsize=(12, 6), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.figure, main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 绑定鼠标事件
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)

    def load_and_display_audio(self):
        """加载并显示音频波形"""
        try:
            # 加载音频文件
            self.audio_data, self.sample_rate = librosa.load(self.audio_file, sr=None)
            total_duration = len(self.audio_data) / self.sample_rate

            # 更新信息
            info_text = f"文件: {os.path.basename(self.audio_file)} | 时长: {total_duration:.1f}秒 | 采样率: {self.sample_rate}Hz"
            info_text += f" | 分割点: {len(self.split_points)}个"
            if self.selected_point_index is not None:
                info_text += f" | 选中: 第{self.selected_point_index + 1}个点"
            info_text += " | 提示: 点击分割点可选中和拖拽"
            self.info_label.config(text=info_text)

            # 绘制波形
            self.draw_waveform()

        except Exception as e:
            self.info_label.config(text=f"加载失败: {str(e)}")

    def draw_waveform(self):
        """绘制音频波形"""
        if self.audio_data is None:
            return

        # 清除之前的图形
        self.figure.clear()

        # 创建子图
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)

        # 时间轴
        time_axis = np.linspace(0, len(self.audio_data) / self.sample_rate, len(self.audio_data))

        # 绘制波形
        ax1.plot(time_axis, self.audio_data, color='blue', alpha=0.7, linewidth=0.5)
        ax1.set_title('音频波形')
        ax1.set_ylabel('振幅')
        ax1.grid(True, alpha=0.3)

        # 分析音量并绘制
        time_points, rms_values = self.splitter.analyze_audio_volume(self.audio_data, self.sample_rate)
        ax2.plot(time_points, rms_values, color='red', linewidth=2, label='RMS音量')
        ax2.set_title('音量变化')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('RMS音量')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 检测并标记静音区域
        silence_regions = self.splitter.find_silence_regions(self.audio_data, self.sample_rate)
        for start, end in silence_regions:
            ax1.axvspan(start, end, alpha=0.3, color='yellow', label='静音区域')
            ax2.axvspan(start, end, alpha=0.3, color='yellow')

        # 标记分割点（如果有）
        if self.split_points:
            for i, point in enumerate(self.split_points):
                # 选中的点用红色，其他用绿色
                color = 'red' if i == self.selected_point_index else 'green'
                linewidth = 3 if i == self.selected_point_index else 2

                ax1.axvline(x=point, color=color, linestyle='--', alpha=0.8, linewidth=linewidth,
                           label=f'分割点{i+1}' if i == 0 else '')
                ax2.axvline(x=point, color=color, linestyle='--', alpha=0.8, linewidth=linewidth)

        # 调整布局
        self.figure.tight_layout()
        self.canvas.draw()

    def set_split_points(self, split_points):
        """设置分割点并重新绘制"""
        self.split_points = split_points
        self.draw_waveform()

    def refresh_waveform(self):
        """刷新波形显示"""
        self.draw_waveform()

    def on_mouse_press(self, event):
        """鼠标按下事件"""
        if event.inaxes is None or self.audio_data is None:
            return

        # 检查是否点击了分割点附近
        click_time = event.xdata
        if click_time is None:
            return

        # 查找最近的分割点
        if self.split_points:
            distances = [abs(point - click_time) for point in self.split_points]
            min_distance = min(distances)

            # 如果点击距离分割点很近（0.2秒内），选中该点
            if min_distance < 0.2:
                self.selected_point_index = distances.index(min_distance)
                self.dragging = True
                self.draw_waveform()
                return

        # 如果没有点击分割点，取消选择
        self.selected_point_index = None
        self.draw_waveform()

    def on_mouse_release(self, event):
        """鼠标释放事件"""
        self.dragging = False

    def on_mouse_motion(self, event):
        """鼠标移动事件"""
        if not self.dragging or self.selected_point_index is None or event.inaxes is None:
            return

        # 拖拽分割点
        new_time = event.xdata
        if new_time is not None and 0 <= new_time <= len(self.audio_data) / self.sample_rate:
            self.split_points[self.selected_point_index] = new_time
            self.draw_waveform()

    def add_split_point(self):
        """添加分割点"""
        if self.audio_data is None:
            return

        # 在音频中间添加一个分割点
        total_duration = len(self.audio_data) / self.sample_rate
        new_point = total_duration / 2

        # 如果已有分割点，在它们之间添加
        if self.split_points:
            self.split_points.append(new_point)
            self.split_points.sort()
        else:
            self.split_points = [new_point]

        self.draw_waveform()

    def delete_selected_point(self):
        """删除选中的分割点"""
        if self.selected_point_index is not None and 0 <= self.selected_point_index < len(self.split_points):
            del self.split_points[self.selected_point_index]
            self.selected_point_index = None
            self.draw_waveform()

    def clear_split_points(self):
        """清除所有分割点"""
        self.split_points = []
        self.selected_point_index = None
        self.draw_waveform()


class AudioSplitterGUI:
    """音频分割工具GUI界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("音频分割工具 v3.2 - 视频时长匹配版")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        self.root.minsize(1200, 700)  # 设置最小窗口大小
        
        # 设置窗口图标（如果有的话）
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        self.splitter = AudioSplitter()
        self.selected_file = ""
        self.is_splitting = False
        self.video_files = []  # 视频文件列表
        self.video_durations = []  # 视频时长列表
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架 - 左右分栏布局
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置主窗口网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)  # 右侧波形区域权重更大
        main_frame.rowconfigure(0, weight=1)

        # 左侧控制面板
        control_panel = ttk.LabelFrame(main_frame, text="控制面板", padding="15")
        control_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))

        # 右侧波形显示区域
        waveform_panel = ttk.LabelFrame(main_frame, text="音频波形可视化", padding="15")
        waveform_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === 左侧控制面板内容 ===

        # 文件选择区域
        file_frame = ttk.LabelFrame(control_panel, text="音频文件选择", padding="12")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        control_panel.columnconfigure(0, weight=1)
        file_frame.columnconfigure(0, weight=1)

        self.file_path_var = tk.StringVar()
        ttk.Label(file_frame, text="选择音频文件:").grid(row=0, column=0, sticky=tk.W, pady=(0, 8))

        file_entry_frame = ttk.Frame(file_frame)
        file_entry_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        file_entry_frame.columnconfigure(0, weight=1)
        file_frame.columnconfigure(0, weight=1)

        self.file_entry = ttk.Entry(file_entry_frame, textvariable=self.file_path_var, state="readonly")
        self.file_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 8))

        ttk.Button(file_entry_frame, text="浏览", command=self.select_file, width=12).grid(row=0, column=1)
        
        # 分割设置区域
        settings_frame = ttk.LabelFrame(control_panel, text="分割设置", padding="12")
        settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        # 分割模式选择
        mode_frame = ttk.Frame(settings_frame)
        mode_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        ttk.Label(mode_frame, text="分割模式:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        self.split_mode_var = tk.StringVar(value="fixed")
        mode_radio_frame = ttk.Frame(mode_frame)
        mode_radio_frame.grid(row=1, column=0, sticky=tk.W)

        self.fixed_radio = ttk.Radiobutton(mode_radio_frame, text="固定时长",
                                         variable=self.split_mode_var, value="fixed",
                                         command=self.on_mode_change)
        self.fixed_radio.grid(row=0, column=0, padx=(0, 20))

        self.custom_radio = ttk.Radiobutton(mode_radio_frame, text="自定义长度",
                                          variable=self.split_mode_var, value="custom",
                                          command=self.on_mode_change)
        self.custom_radio.grid(row=0, column=1, padx=(0, 20))

        self.video_radio = ttk.Radiobutton(mode_radio_frame, text="视频时长匹配",
                                         variable=self.split_mode_var, value="video",
                                         command=self.on_mode_change)
        self.video_radio.grid(row=0, column=2)

        # 固定时长设置
        self.fixed_frame = ttk.Frame(settings_frame)
        self.fixed_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(self.fixed_frame, text="分割时长:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        duration_frame = ttk.Frame(self.fixed_frame)
        duration_frame.grid(row=1, column=0, sticky=tk.W)

        self.duration_var = tk.StringVar(value="60")
        self.duration_entry = ttk.Entry(duration_frame, textvariable=self.duration_var, width=10)
        self.duration_entry.grid(row=0, column=0, padx=(0, 10))

        self.time_unit_var = tk.StringVar(value="秒")
        time_unit_combo = ttk.Combobox(duration_frame, textvariable=self.time_unit_var,
                                     values=["秒", "分钟"], state="readonly", width=8)
        time_unit_combo.grid(row=0, column=1)

        # 自定义长度设置
        self.custom_frame = ttk.Frame(settings_frame)
        self.custom_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(self.custom_frame, text="自定义长度:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        custom_input_frame = ttk.Frame(self.custom_frame)
        custom_input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        custom_input_frame.columnconfigure(0, weight=1)

        self.custom_durations_var = tk.StringVar()
        self.custom_entry = ttk.Entry(custom_input_frame, textvariable=self.custom_durations_var)
        self.custom_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.custom_entry.bind('<KeyRelease>', self.on_custom_input_change)

        ttk.Label(custom_input_frame, text="秒").grid(row=0, column=1)

        # 说明文本
        help_text = "请输入每段的长度，用逗号分隔，如：3,5,10"
        self.help_label = ttk.Label(self.custom_frame, text=help_text,
                                   foreground="gray", font=("TkDefaultFont", 8))
        self.help_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))

        # 预览信息
        self.preview_label = ttk.Label(self.custom_frame, text="",
                                     foreground="blue", font=("TkDefaultFont", 8))
        self.preview_label.grid(row=3, column=0, sticky=tk.W, pady=(5, 0))

        # 视频时长匹配设置
        self.video_frame = ttk.Frame(settings_frame)
        self.video_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(self.video_frame, text="视频文件:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        video_button_frame = ttk.Frame(self.video_frame)
        video_button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        video_button_frame.columnconfigure(0, weight=1)

        self.select_videos_button = ttk.Button(video_button_frame, text="📁 选择视频文件",
                                             command=self.select_video_files, width=18)
        self.select_videos_button.grid(row=0, column=0, sticky=tk.W, padx=(0, 12))

        self.clear_videos_button = ttk.Button(video_button_frame, text="🗑️ 清空列表",
                                            command=self.clear_video_files, width=15)
        self.clear_videos_button.grid(row=0, column=1, sticky=tk.W)

        # 视频列表显示
        video_list_frame = ttk.Frame(self.video_frame)
        video_list_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        video_list_frame.columnconfigure(0, weight=1)

        # 创建Treeview来显示视频列表
        columns = ('序号', '文件名', '时长')
        self.video_tree = ttk.Treeview(video_list_frame, columns=columns, show='headings', height=7)

        # 设置列标题
        self.video_tree.heading('序号', text='序号', anchor='center')
        self.video_tree.heading('文件名', text='文件名', anchor='w')
        self.video_tree.heading('时长', text='时长(秒)', anchor='center')

        # 设置列宽和样式
        self.video_tree.column('序号', width=60, anchor='center', minwidth=50)
        self.video_tree.column('文件名', width=220, anchor='w', minwidth=150)
        self.video_tree.column('时长', width=120, anchor='center', minwidth=80)

        self.video_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 添加滚动条
        video_scrollbar = ttk.Scrollbar(video_list_frame, orient=tk.VERTICAL, command=self.video_tree.yview)
        video_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.video_tree.configure(yscrollcommand=video_scrollbar.set)

        # 视频列表操作按钮
        video_ops_frame = ttk.Frame(self.video_frame)
        video_ops_frame.grid(row=3, column=0, sticky=tk.W, pady=(12, 0))

        self.move_up_button = ttk.Button(video_ops_frame, text="↑ 上移", command=self.move_video_up, width=10)
        self.move_up_button.grid(row=0, column=0, padx=(0, 8))

        self.move_down_button = ttk.Button(video_ops_frame, text="↓ 下移", command=self.move_video_down, width=10)
        self.move_down_button.grid(row=0, column=1, padx=(0, 8))

        self.remove_video_button = ttk.Button(video_ops_frame, text="✕ 删除", command=self.remove_selected_video, width=10)
        self.remove_video_button.grid(row=0, column=2)

        # 视频匹配信息
        self.video_info_label = ttk.Label(self.video_frame, text="",
                                        foreground="blue", font=("TkDefaultFont", 8))
        self.video_info_label.grid(row=4, column=0, sticky=tk.W, pady=(10, 0))

        # 智能分割选项
        smart_split_frame = ttk.LabelFrame(settings_frame, text="智能分割设置", padding="12")
        smart_split_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        settings_frame.columnconfigure(0, weight=1)
        smart_split_frame.columnconfigure(0, weight=1)

        self.smart_split_var = tk.BooleanVar(value=False)
        self.smart_split_check = ttk.Checkbutton(smart_split_frame,
                                                text="启用智能分割（在音频低谷处分割，避免卡顿感）",
                                                variable=self.smart_split_var,
                                                command=self.on_smart_split_change)
        self.smart_split_check.grid(row=0, column=0, sticky=tk.W, pady=(0, 8))

        # 搜索范围设置
        range_frame = ttk.Frame(smart_split_frame)
        range_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

        ttk.Label(range_frame, text="搜索范围:").grid(row=0, column=0, sticky=tk.W, padx=(20, 5))

        self.search_range_var = tk.StringVar(value="2.0")
        self.search_range_entry = ttk.Entry(range_frame, textvariable=self.search_range_var, width=8)
        self.search_range_entry.grid(row=0, column=1, padx=(0, 5))

        ttk.Label(range_frame, text="秒 (建议1-5秒)").grid(row=0, column=2, sticky=tk.W)

        # 智能分割说明
        help_text = "智能分割会在目标时间点前后指定范围内寻找音量最低的位置进行分割"
        self.smart_help_label = ttk.Label(smart_split_frame, text=help_text,
                                         foreground="gray", font=("TkDefaultFont", 8))
        self.smart_help_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        
        # 操作按钮区域
        button_frame = ttk.LabelFrame(control_panel, text="操作", padding="12")
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        # 按钮网格布局
        self.split_button = ttk.Button(button_frame, text="🎵 开始分割", command=self.start_splitting)
        self.split_button.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.refresh_waveform_button = ttk.Button(button_frame, text="🔄 刷新波形", command=self.refresh_waveform)
        self.refresh_waveform_button.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 8))

        ttk.Button(button_frame, text="❌ 退出", command=self.root.quit).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(8, 0))

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        # 进度显示区域
        progress_frame = ttk.LabelFrame(control_panel, text="状态信息", padding="12")
        progress_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_panel.rowconfigure(3, weight=1)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, style="TProgressbar")
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.status_var = tk.StringVar(value="🎵 请选择音频文件并设置分割参数")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var, wraplength=350,
                                    font=("TkDefaultFont", 9), foreground="#2E86AB")
        self.status_label.grid(row=1, column=0, sticky=(tk.W, tk.E))

        progress_frame.columnconfigure(0, weight=1)
        
        # === 右侧波形显示区域 ===

        # 波形信息标签
        self.waveform_info_var = tk.StringVar(value="请选择音频文件以显示波形")
        self.waveform_info_label = ttk.Label(waveform_panel, textvariable=self.waveform_info_var,
                                           font=("TkDefaultFont", 9))
        self.waveform_info_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # 波形图区域
        self.figure = plt.Figure(figsize=(10, 6), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.figure, waveform_panel)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置波形面板网格权重
        waveform_panel.columnconfigure(0, weight=1)
        waveform_panel.rowconfigure(1, weight=1)

        # 绑定鼠标事件
        self.canvas.mpl_connect('button_press_event', self.on_waveform_click)

        # 初始化波形相关变量
        self.audio_data = None
        self.sample_rate = None
        self.split_points = []
        self.selected_point_index = None

        # 初始化界面状态
        self.on_mode_change()
        self.draw_empty_waveform()
        
    def select_file(self):
        """选择音频文件"""
        file_types = [
            ("音频文件", "*.mp3 *.wav"),
            ("MP3文件", "*.mp3"),
            ("WAV文件", "*.wav"),
            ("所有文件", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=file_types
        )
        
        if filename:
            self.selected_file = filename
            self.file_path_var.set(filename)
            self.status_var.set(f"✅ 已选择文件: {os.path.basename(filename)}")
            # 更新自定义长度预览
            self.update_custom_preview()
            # 加载并显示波形
            self.load_and_display_waveform()
    
    def get_duration_in_seconds(self):
        """获取分割时长（转换为秒）"""
        try:
            duration = float(self.duration_var.get())
            if self.time_unit_var.get() == "分钟":
                duration *= 60
            return duration  # 保持浮点数精度，不使用int()截断
        except ValueError:
            return None

    def on_mode_change(self):
        """分割模式切换事件处理"""
        mode = self.split_mode_var.get()
        if mode == "fixed":
            # 显示固定时长设置，隐藏其他设置
            self.fixed_frame.grid()
            self.custom_frame.grid_remove()
            self.video_frame.grid_remove()
        elif mode == "custom":
            # 显示自定义长度设置，隐藏其他设置
            self.fixed_frame.grid_remove()
            self.custom_frame.grid()
            self.video_frame.grid_remove()
            self.update_custom_preview()
        elif mode == "video":
            # 显示视频时长匹配设置，隐藏其他设置
            self.fixed_frame.grid_remove()
            self.custom_frame.grid_remove()
            self.video_frame.grid()
            self.update_video_info()

        # 更新波形分割点
        self.update_waveform_split_points()

    def on_custom_input_change(self, event=None):
        """自定义长度输入变化事件处理"""
        self.update_custom_preview()
        # 更新波形分割点
        self.update_waveform_split_points()

    def parse_custom_durations(self, input_str):
        """解析自定义长度字符串"""
        if not input_str.strip():
            return None, "请输入自定义长度"

        try:
            # 分割字符串并转换为数字
            parts = [part.strip() for part in input_str.split(',')]
            durations = []

            for i, part in enumerate(parts):
                if not part:
                    return None, f"第{i+1}个长度不能为空"

                try:
                    duration = float(part)
                    if duration <= 0:
                        return None, f"第{i+1}个长度必须大于0"
                    durations.append(duration)
                except ValueError:
                    return None, f"第{i+1}个长度格式错误: {part}"

            if len(durations) == 0:
                return None, "至少需要输入一个长度"

            return durations, None

        except Exception as e:
            return None, f"解析错误: {str(e)}"

    def select_video_files(self):
        """选择视频文件"""
        file_types = [
            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v"),
            ("MP4文件", "*.mp4"),
            ("AVI文件", "*.avi"),
            ("MOV文件", "*.mov"),
            ("所有文件", "*.*")
        ]

        files = filedialog.askopenfilenames(
            title="选择视频文件",
            filetypes=file_types
        )

        if files:
            self.add_video_files(files)

    def add_video_files(self, file_paths):
        """添加视频文件到列表"""
        # 验证文件
        valid_files, invalid_files, error_messages = self.splitter.video_processor.validate_video_files(file_paths)

        if invalid_files:
            error_msg = "以下文件无法添加:\n" + "\n".join(error_messages)
            messagebox.showwarning("文件验证", error_msg)

        if valid_files:
            # 读取视频时长
            self.read_video_durations(valid_files)

    def read_video_durations(self, file_paths):
        """读取视频文件时长"""
        def progress_callback(progress, message):
            # 这里可以添加进度显示
            pass

        try:
            # 批量读取视频时长
            results = self.splitter.video_processor.batch_get_video_durations(file_paths, progress_callback)

            # 添加到列表
            for result in results:
                if result['success']:
                    self.video_files.append(result['file_path'])
                    self.video_durations.append(result['duration'])

            # 更新界面显示
            self.update_video_list_display()
            self.update_video_info()

            if len(results) > 0:
                success_count = sum(1 for r in results if r['success'])
                messagebox.showinfo("读取完成", f"成功读取 {success_count}/{len(results)} 个视频文件的时长")

        except Exception as e:
            messagebox.showerror("错误", f"读取视频时长时发生错误: {str(e)}")

    def update_video_list_display(self):
        """更新视频列表显示"""
        # 清空现有项目
        for item in self.video_tree.get_children():
            self.video_tree.delete(item)

        # 添加视频文件信息
        for i, (file_path, duration) in enumerate(zip(self.video_files, self.video_durations)):
            file_name = os.path.basename(file_path)
            duration_str = f"{duration:.3f}"
            self.video_tree.insert('', 'end', values=(i+1, file_name, duration_str))

    def update_video_info(self):
        """更新视频匹配信息"""
        if not self.video_files:
            self.video_info_label.config(text="📹 请选择视频文件", foreground="#95A5A6")
            return

        total_duration = sum(self.video_durations)
        file_count = len(self.video_files)

        info_text = f"📹 共 {file_count} 个视频文件，总时长: {total_duration:.3f}秒"
        self.video_info_label.config(text=info_text, foreground="#2E86AB")

    def clear_video_files(self):
        """清空视频文件列表"""
        self.video_files.clear()
        self.video_durations.clear()
        self.update_video_list_display()
        self.update_video_info()
        self.update_waveform_split_points()

    def move_video_up(self):
        """上移选中的视频"""
        selection = self.video_tree.selection()
        if not selection:
            return

        item = selection[0]
        index = self.video_tree.index(item)

        if index > 0:
            # 交换列表中的位置
            self.video_files[index], self.video_files[index-1] = self.video_files[index-1], self.video_files[index]
            self.video_durations[index], self.video_durations[index-1] = self.video_durations[index-1], self.video_durations[index]

            # 更新显示
            self.update_video_list_display()

            # 重新选中移动后的项目
            new_item = self.video_tree.get_children()[index-1]
            self.video_tree.selection_set(new_item)

            # 更新分割点
            self.update_waveform_split_points()

    def move_video_down(self):
        """下移选中的视频"""
        selection = self.video_tree.selection()
        if not selection:
            return

        item = selection[0]
        index = self.video_tree.index(item)

        if index < len(self.video_files) - 1:
            # 交换列表中的位置
            self.video_files[index], self.video_files[index+1] = self.video_files[index+1], self.video_files[index]
            self.video_durations[index], self.video_durations[index+1] = self.video_durations[index+1], self.video_durations[index]

            # 更新显示
            self.update_video_list_display()

            # 重新选中移动后的项目
            new_item = self.video_tree.get_children()[index+1]
            self.video_tree.selection_set(new_item)

            # 更新分割点
            self.update_waveform_split_points()

    def remove_selected_video(self):
        """删除选中的视频"""
        selection = self.video_tree.selection()
        if not selection:
            return

        item = selection[0]
        index = self.video_tree.index(item)

        # 从列表中删除
        del self.video_files[index]
        del self.video_durations[index]

        # 更新显示
        self.update_video_list_display()
        self.update_video_info()
        self.update_waveform_split_points()

    def update_custom_preview(self):
        """更新自定义长度预览信息"""
        if self.split_mode_var.get() != "custom":
            return

        input_str = self.custom_durations_var.get()
        durations, error = self.parse_custom_durations(input_str)

        if error:
            self.preview_label.config(text=f"❌ {error}", foreground="#E74C3C")
            return

        if not durations:
            self.preview_label.config(text="", foreground="blue")
            return

        # 计算总时长
        total_specified = sum(durations)

        # 如果有选择的文件，计算最后一段长度
        if self.selected_file:
            try:
                # 这里简化处理，实际应用中可能需要缓存音频时长
                import librosa
                audio_data, sample_rate = librosa.load(self.selected_file, sr=None)
                total_duration = len(audio_data) / sample_rate

                last_segment = total_duration - total_specified

                if total_specified >= total_duration:
                    self.preview_label.config(
                        text=f"❌ 指定总时长({total_specified:.1f}秒)超出音频时长({total_duration:.1f}秒)",
                        foreground="#E74C3C"
                    )
                elif last_segment < 1.0:
                    self.preview_label.config(
                        text=f"⚠️ 将生成{len(durations)}段，最后一段({last_segment:.1f}秒)将被跳过",
                        foreground="#F39C12"
                    )
                else:
                    self.preview_label.config(
                        text=f"✅ 将生成{len(durations)+1}段，最后一段{last_segment:.1f}秒",
                        foreground="#27AE60"
                    )
            except:
                self.preview_label.config(
                    text=f"ℹ️ 指定{len(durations)}段，总时长{total_specified:.1f}秒",
                    foreground="#2E86AB"
                )
        else:
            self.preview_label.config(
                text=f"ℹ️ 指定{len(durations)}段，总时长{total_specified:.1f}秒",
                foreground="#2E86AB"
            )

    def on_smart_split_change(self):
        """智能分割选项变化事件"""
        # 当智能分割选项变化时，重新计算和显示分割点
        if self.audio_data is not None:
            self.update_waveform_split_points()

    def get_search_range(self):
        """获取搜索范围设置"""
        try:
            range_value = float(self.search_range_var.get())
            return max(0.5, min(5.0, range_value))  # 限制在0.5-5秒之间
        except ValueError:
            return 2.0  # 默认值

    def draw_empty_waveform(self):
        """绘制空的波形图"""
        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, '请选择音频文件以显示波形',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.canvas.draw()

    def load_and_display_waveform(self):
        """加载并显示音频波形"""
        if not self.selected_file:
            return

        try:
            # 加载音频文件
            self.audio_data, self.sample_rate = librosa.load(self.selected_file, sr=None)
            total_duration = len(self.audio_data) / self.sample_rate

            # 更新信息
            info_text = f"文件: {os.path.basename(self.selected_file)} | "
            info_text += f"时长: {total_duration:.1f}秒 | 采样率: {self.sample_rate}Hz"
            self.waveform_info_var.set(info_text)

            # 绘制波形
            self.draw_waveform()

        except Exception as e:
            self.waveform_info_var.set(f"加载失败: {str(e)}")
            self.draw_empty_waveform()

    def draw_waveform(self):
        """绘制音频波形"""
        if self.audio_data is None:
            self.draw_empty_waveform()
            return

        # 清除之前的图形
        self.figure.clear()

        # 创建子图
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)

        # 时间轴
        time_axis = np.linspace(0, len(self.audio_data) / self.sample_rate, len(self.audio_data))

        # 绘制波形（采样显示以提高性能）
        step = max(1, len(self.audio_data) // 10000)  # 最多显示10000个点
        ax1.plot(time_axis[::step], self.audio_data[::step], color='#2E86AB', alpha=0.8, linewidth=0.6)
        ax1.set_title('🎵 音频波形', fontsize=12, fontweight='bold', color='#2E86AB')
        ax1.set_ylabel('振幅', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.set_facecolor('#F8F9FA')

        # 分析音量并绘制
        time_points, rms_values = self.splitter.analyze_audio_volume(self.audio_data, self.sample_rate)
        ax2.plot(time_points, rms_values, color='#E74C3C', linewidth=2.5, label='RMS音量', alpha=0.9)
        ax2.set_title('📊 音量变化', fontsize=12, fontweight='bold', color='#E74C3C')
        ax2.set_xlabel('时间 (秒)', fontsize=10)
        ax2.set_ylabel('RMS音量', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.set_facecolor('#F8F9FA')
        ax2.legend(loc='upper right')

        # 检测并标记静音区域
        silence_regions = self.splitter.find_silence_regions(self.audio_data, self.sample_rate)
        for start, end in silence_regions:
            ax1.axvspan(start, end, alpha=0.2, color='#F39C12', label='🔇 静音区域' if start == silence_regions[0][0] else '')
            ax2.axvspan(start, end, alpha=0.2, color='#F39C12')

        # 标记分割点
        self.draw_split_points(ax1, ax2)

        # 调整布局
        self.figure.tight_layout()
        self.canvas.draw()

    def draw_split_points(self, ax1, ax2):
        """在波形图上绘制分割点"""
        if not self.split_points:
            return

        for i, point in enumerate(self.split_points):
            # 选中的点用红色，其他用绿色
            color = '#E74C3C' if i == self.selected_point_index else '#27AE60'
            linewidth = 3.5 if i == self.selected_point_index else 2.5
            alpha = 0.9 if i == self.selected_point_index else 0.7

            ax1.axvline(x=point, color=color, linestyle='--', alpha=alpha, linewidth=linewidth,
                       label=f'✂️ 分割点' if i == 0 else '')
            ax2.axvline(x=point, color=color, linestyle='--', alpha=alpha, linewidth=linewidth)

    def update_waveform_split_points(self):
        """根据当前设置更新波形上的分割点"""
        if self.audio_data is None:
            return

        mode = self.split_mode_var.get()
        smart_split = self.smart_split_var.get()
        search_range = self.get_search_range()

        self.split_points = []

        try:
            if mode == "fixed":
                duration = self.get_duration_in_seconds()
                if duration and duration > 0:
                    # 计算固定时长分割点
                    total_duration = len(self.audio_data) / self.sample_rate
                    num_segments = math.ceil(total_duration / duration)

                    if smart_split:
                        # 智能分割点
                        current_position = 0
                        self.split_points.append(current_position)

                        for i in range(1, num_segments):
                            target_time = current_position + duration
                            optimal_time = self.splitter.find_optimal_split_point(
                                self.audio_data, self.sample_rate, target_time, search_range)
                            self.split_points.append(optimal_time)
                            current_position = optimal_time
                    else:
                        # 高精度固定分割点
                        for i in range(1, num_segments):
                            target_time = current_position + duration
                            # 使用round()确保精确对齐到采样点
                            target_sample = round(target_time * self.sample_rate)
                            precise_time = target_sample / self.sample_rate
                            self.split_points.append(min(precise_time, total_duration))
                            current_position = precise_time

            elif mode == "custom":
                durations, error = self.parse_custom_durations(self.custom_durations_var.get())
                if durations and not error:
                    # 计算自定义长度分割点
                    current_position = 0
                    self.split_points.append(current_position)

                    for duration in durations:
                        target_time = current_position + duration
                        if smart_split:
                            optimal_time = self.splitter.find_optimal_split_point(
                                self.audio_data, self.sample_rate, target_time, search_range)
                            self.split_points.append(optimal_time)
                            current_position = optimal_time
                        else:
                            # 高精度固定分割：确保分割点对齐到采样点边界
                            target_sample = round(target_time * self.sample_rate)
                            precise_time = target_sample / self.sample_rate
                            self.split_points.append(precise_time)
                            current_position = precise_time

            elif mode == "video":
                if self.video_durations:
                    # 计算视频时长匹配分割点
                    current_position = 0
                    self.split_points.append(current_position)

                    for duration in self.video_durations:
                        target_time = current_position + duration
                        if smart_split:
                            optimal_time = self.splitter.find_optimal_split_point(
                                self.audio_data, self.sample_rate, target_time, search_range)
                            self.split_points.append(optimal_time)
                            current_position = optimal_time
                        else:
                            # 高精度固定分割：确保分割点对齐到采样点边界
                            target_sample = round(target_time * self.sample_rate)
                            precise_time = target_sample / self.sample_rate
                            self.split_points.append(precise_time)
                            current_position = precise_time

            # 重新绘制波形
            self.draw_waveform()

        except Exception as e:
            print(f"更新分割点时发生错误: {e}")

    def on_waveform_click(self, event):
        """波形图点击事件"""
        if event.inaxes is None or self.audio_data is None:
            return

        click_time = event.xdata
        if click_time is None:
            return

        # 查找最近的分割点
        if self.split_points:
            distances = [abs(point - click_time) for point in self.split_points]
            min_distance = min(distances)

            # 如果点击距离分割点很近（0.3秒内），选中该点
            if min_distance < 0.3:
                self.selected_point_index = distances.index(min_distance)
                self.draw_waveform()
                return

        # 如果没有点击分割点，取消选择
        self.selected_point_index = None
        self.draw_waveform()

    def refresh_waveform(self):
        """刷新波形显示"""
        if self.selected_file:
            self.load_and_display_waveform()
            self.update_waveform_split_points()
        else:
            self.draw_empty_waveform()


    
    def update_progress(self, progress, message):
        """更新进度显示"""
        self.progress_var.set(progress)
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def start_splitting(self):
        """开始分割音频"""
        if self.is_splitting:
            return

        # 验证输入
        if not self.selected_file:
            messagebox.showerror("错误", "请先选择音频文件")
            return

        mode = self.split_mode_var.get()
        smart_split = self.smart_split_var.get()
        search_range = self.get_search_range()

        if mode == "fixed":
            # 固定时长模式
            duration = self.get_duration_in_seconds()
            if duration is None or duration <= 0:
                messagebox.showerror("错误", "请输入有效的分割时长")
                return

            # 禁用分割按钮
            self.is_splitting = True
            self.split_button.config(state="disabled")

            # 在新线程中执行分割操作
            thread = threading.Thread(target=self.split_audio_thread, args=(duration, None, smart_split, search_range))
            thread.daemon = True
            thread.start()

        elif mode == "custom":
            # 自定义长度模式
            durations, error = self.parse_custom_durations(self.custom_durations_var.get())
            if error:
                messagebox.showerror("错误", error)
                return

            if not durations:
                messagebox.showerror("错误", "请输入有效的自定义长度")
                return

            # 禁用分割按钮
            self.is_splitting = True
            self.split_button.config(state="disabled")

            # 在新线程中执行分割操作
            thread = threading.Thread(target=self.split_audio_thread, args=(None, durations, smart_split, search_range))
            thread.daemon = True
            thread.start()

        elif mode == "video":
            # 视频时长匹配模式
            if not self.video_durations:
                messagebox.showerror("错误", "请先选择视频文件")
                return

            # 禁用分割按钮
            self.is_splitting = True
            self.split_button.config(state="disabled")

            # 在新线程中执行分割操作
            thread = threading.Thread(target=self.split_audio_thread, args=(None, self.video_durations, smart_split, search_range))
            thread.daemon = True
            thread.start()
    
    def split_audio_thread(self, duration, custom_durations, smart_split, search_range):
        """在后台线程中执行音频分割"""
        try:
            success, message, output_files = self.splitter.split_audio(
                self.selected_file,
                segment_duration=duration,
                custom_durations=custom_durations,
                smart_split=smart_split,
                search_range=search_range,
                progress_callback=self.update_progress
            )

            # 在主线程中显示结果
            self.root.after(0, self.show_result, success, message, output_files)

        except Exception as e:
            self.root.after(0, self.show_result, False, f"分割过程中发生错误: {str(e)}", [])
    
    def show_result(self, success, message, output_files):
        """显示分割结果"""
        self.is_splitting = False
        self.split_button.config(state="normal")
        
        if success:
            self.progress_var.set(100)
            self.status_var.set(message)
            messagebox.showinfo("成功", f"{message}\n\n输出目录: {os.path.dirname(output_files[0]) if output_files else ''}")
        else:
            self.progress_var.set(0)
            self.status_var.set(f"分割失败: {message}")
            messagebox.showerror("错误", message)
    
    def run(self):
        """运行GUI应用"""
        self.root.mainloop()


def main():
    """主函数"""
    app = AudioSplitterGUI()
    app.run()


if __name__ == "__main__":
    main()
