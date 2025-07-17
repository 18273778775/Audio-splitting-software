#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI优化后的功能测试脚本
测试所有功能是否正常工作，包括视频时长匹配功能
"""

import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path
import time

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import AudioSplitter
from video_processor import VideoProcessor


class ComprehensiveTester:
    """综合功能测试类"""
    
    def __init__(self):
        self.splitter = AudioSplitter()
        self.video_processor = VideoProcessor()
        self.test_results = []
        
    def create_test_audio(self, duration=20.0, sample_rate=44100):
        """创建测试音频文件"""
        print(f"[INFO] 创建测试音频文件 (时长: {duration}秒, 采样率: {sample_rate}Hz)")
        
        # 生成复合音频信号（多个频率的正弦波）
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        
        # 基础频率
        freq1 = 440  # A4
        freq2 = 880  # A5
        freq3 = 220  # A3
        
        # 创建复合信号
        audio_data = (0.3 * np.sin(2 * np.pi * freq1 * t) + 
                     0.2 * np.sin(2 * np.pi * freq2 * t) + 
                     0.1 * np.sin(2 * np.pi * freq3 * t))
        
        # 添加一些静音段来测试智能分割
        silence_start = int(8 * sample_rate)
        silence_end = int(9 * sample_rate)
        audio_data[silence_start:silence_end] *= 0.01  # 几乎静音
        
        # 保存为WAV文件
        test_file = "comprehensive_test_audio.wav"
        sf.write(test_file, audio_data, sample_rate)
        
        return test_file, audio_data, sample_rate
    
    def test_basic_audio_loading(self, test_file):
        """测试基础音频加载功能"""
        print("\n[TEST 1] 测试基础音频加载功能")
        print("-" * 50)
        
        try:
            # 测试音频格式检查
            is_supported = self.splitter.is_supported_format(test_file)
            print(f"音频格式支持检查: {'通过' if is_supported else '失败'}")
            
            if not is_supported:
                return False
            
            # 测试音频加载
            import librosa
            audio_data, sample_rate = librosa.load(test_file, sr=None)
            print(f"音频加载成功: 时长={len(audio_data)/sample_rate:.3f}秒, 采样率={sample_rate}Hz")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 基础音频加载测试失败: {e}")
            return False
    
    def test_fixed_duration_split(self, test_file):
        """测试固定时长分割功能"""
        print("\n[TEST 2] 测试固定时长分割功能")
        print("-" * 50)
        
        try:
            # 测试固定时长分割
            duration = 3.5
            print(f"测试固定时长分割: {duration}秒")
            
            success, message, output_files = self.splitter.split_audio(
                test_file, segment_duration=duration, smart_split=False
            )
            
            if success:
                print(f"[OK] 分割成功: {message}")
                print(f"生成文件数量: {len(output_files)}")
                
                # 验证精确度
                total_error = 0
                for i, file_path in enumerate(output_files):
                    if os.path.exists(file_path):
                        audio_data, sr = sf.read(file_path)
                        actual_duration = len(audio_data) / sr
                        expected_duration = duration if i < len(output_files) - 1 else None
                        
                        if expected_duration:
                            error = abs(actual_duration - expected_duration)
                            total_error += error
                            print(f"  片段 {i+1}: 期望={expected_duration:.6f}s, 实际={actual_duration:.6f}s, 误差={error*1000:.3f}ms")
                        else:
                            print(f"  片段 {i+1}: 最后片段={actual_duration:.6f}s")
                        
                        # 清理文件
                        os.remove(file_path)
                
                # 清理输出目录
                output_dir = Path("output")
                if output_dir.exists() and not any(output_dir.iterdir()):
                    output_dir.rmdir()
                
                avg_error_ms = (total_error / max(1, len(output_files) - 1)) * 1000
                print(f"平均误差: {avg_error_ms:.3f}ms")
                
                return avg_error_ms <= 1.0
            else:
                print(f"[FAIL] 分割失败: {message}")
                return False
                
        except Exception as e:
            print(f"[ERROR] 固定时长分割测试失败: {e}")
            return False
    
    def test_custom_duration_split(self, test_file):
        """测试自定义时长分割功能"""
        print("\n[TEST 3] 测试自定义时长分割功能")
        print("-" * 50)
        
        try:
            # 测试自定义时长分割
            custom_durations = [2.5, 4.0, 3.2, 2.8]
            print(f"测试自定义时长分割: {custom_durations}")
            
            success, message, output_files = self.splitter.split_audio(
                test_file, custom_durations=custom_durations, smart_split=False
            )
            
            if success:
                print(f"[OK] 分割成功: {message}")
                print(f"生成文件数量: {len(output_files)}")
                
                # 验证精确度
                total_error = 0
                for i, file_path in enumerate(output_files):
                    if os.path.exists(file_path) and i < len(custom_durations):
                        audio_data, sr = sf.read(file_path)
                        actual_duration = len(audio_data) / sr
                        expected_duration = custom_durations[i]
                        error = abs(actual_duration - expected_duration)
                        total_error += error
                        print(f"  片段 {i+1}: 期望={expected_duration:.6f}s, 实际={actual_duration:.6f}s, 误差={error*1000:.3f}ms")
                        
                        # 清理文件
                        os.remove(file_path)
                
                # 清理输出目录
                output_dir = Path("output")
                if output_dir.exists() and not any(output_dir.iterdir()):
                    output_dir.rmdir()
                
                avg_error_ms = (total_error / len(custom_durations)) * 1000
                print(f"平均误差: {avg_error_ms:.3f}ms")
                
                return avg_error_ms <= 1.0
            else:
                print(f"[FAIL] 分割失败: {message}")
                return False
                
        except Exception as e:
            print(f"[ERROR] 自定义时长分割测试失败: {e}")
            return False
    
    def test_smart_split(self, test_file):
        """测试智能分割功能"""
        print("\n[TEST 4] 测试智能分割功能")
        print("-" * 50)
        
        try:
            # 测试智能分割
            duration = 4.0
            search_range = 1.0
            print(f"测试智能分割: 目标时长={duration}秒, 搜索范围={search_range}秒")
            
            success, message, output_files = self.splitter.split_audio(
                test_file, segment_duration=duration, smart_split=True, search_range=search_range
            )
            
            if success:
                print(f"[OK] 智能分割成功: {message}")
                print(f"生成文件数量: {len(output_files)}")
                
                # 验证连续性
                reconstructed_audio = []
                for i, file_path in enumerate(output_files):
                    if os.path.exists(file_path):
                        audio_data, sr = sf.read(file_path)
                        reconstructed_audio.extend(audio_data)
                        actual_duration = len(audio_data) / sr
                        print(f"  片段 {i+1}: 时长={actual_duration:.6f}s")
                        
                        # 清理文件
                        os.remove(file_path)
                
                # 清理输出目录
                output_dir = Path("output")
                if output_dir.exists() and not any(output_dir.iterdir()):
                    output_dir.rmdir()
                
                print(f"重构音频长度: {len(reconstructed_audio)} 采样点")
                return True
            else:
                print(f"[FAIL] 智能分割失败: {message}")
                return False
                
        except Exception as e:
            print(f"[ERROR] 智能分割测试失败: {e}")
            return False
    
    def test_video_duration_matching(self, test_file):
        """测试视频时长匹配功能"""
        print("\n[TEST 5] 测试视频时长匹配功能")
        print("-" * 50)
        
        try:
            # 模拟视频时长列表
            video_durations = [3.123, 4.567, 2.890, 5.234]
            print(f"测试视频时长匹配: {video_durations}")
            
            success, message, output_files = self.splitter.split_audio_by_video_durations(
                test_file, video_durations, smart_split=False
            )
            
            if success:
                print(f"[OK] 视频时长匹配分割成功: {message}")
                print(f"生成文件数量: {len(output_files)}")
                
                # 验证精确度
                total_error = 0
                for i, file_path in enumerate(output_files):
                    if os.path.exists(file_path) and i < len(video_durations):
                        audio_data, sr = sf.read(file_path)
                        actual_duration = len(audio_data) / sr
                        expected_duration = video_durations[i]
                        error = abs(actual_duration - expected_duration)
                        total_error += error
                        print(f"  片段 {i+1}: 期望={expected_duration:.6f}s, 实际={actual_duration:.6f}s, 误差={error*1000:.3f}ms")
                        
                        # 清理文件
                        os.remove(file_path)
                
                # 清理输出目录
                output_dir = Path("output")
                if output_dir.exists() and not any(output_dir.iterdir()):
                    output_dir.rmdir()
                
                avg_error_ms = (total_error / len(video_durations)) * 1000
                print(f"平均误差: {avg_error_ms:.3f}ms")
                
                return avg_error_ms <= 1.0
            else:
                print(f"[FAIL] 视频时长匹配分割失败: {message}")
                return False
                
        except Exception as e:
            print(f"[ERROR] 视频时长匹配测试失败: {e}")
            return False
    
    def test_video_processor(self):
        """测试视频处理器功能"""
        print("\n[TEST 6] 测试视频处理器功能")
        print("-" * 50)
        
        try:
            # 测试格式支持检查
            test_formats = [
                ("test.mp4", True),
                ("test.avi", True),
                ("test.mov", True),
                ("test.txt", False),
                ("test.mp3", False)
            ]
            
            format_test_passed = True
            for filename, expected in test_formats:
                result = self.video_processor.is_supported_format(filename)
                status = "通过" if result == expected else "失败"
                print(f"  格式检查 {filename}: {status}")
                if result != expected:
                    format_test_passed = False
            
            # 测试时长格式化
            test_durations = [0.0, 30.5, 65.123, 3661.789]
            for duration in test_durations:
                formatted = self.video_processor.format_duration(duration)
                print(f"  时长格式化 {duration:.3f}s -> {formatted}")
            
            print(f"视频处理器功能测试: {'通过' if format_test_passed else '失败'}")
            return format_test_passed
            
        except Exception as e:
            print(f"[ERROR] 视频处理器测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("UI优化后的综合功能测试")
        print("=" * 60)
        
        # 创建测试音频
        test_file, original_audio, sample_rate = self.create_test_audio()
        
        # 运行所有测试
        tests = [
            ("基础音频加载", lambda: self.test_basic_audio_loading(test_file)),
            ("固定时长分割", lambda: self.test_fixed_duration_split(test_file)),
            ("自定义时长分割", lambda: self.test_custom_duration_split(test_file)),
            ("智能分割", lambda: self.test_smart_split(test_file)),
            ("视频时长匹配", lambda: self.test_video_duration_matching(test_file)),
            ("视频处理器", lambda: self.test_video_processor())
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"[ERROR] 测试 {test_name} 发生异常: {e}")
                results.append((test_name, False))
        
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
        
        # 生成测试报告
        self.generate_test_report(results)
        
        return results
    
    def generate_test_report(self, results):
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("测试结果报告")
        print("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "[PASS]" if result else "[FAIL]"
            print(f"{status} {test_name}")
            if result:
                passed += 1
        
        print(f"\n总体结果: {passed}/{total} 测试通过")
        success_rate = (passed / total) * 100
        print(f"成功率: {success_rate:.1f}%")
        
        if passed == total:
            print("\n[SUCCESS] 所有功能测试通过！UI优化成功，所有功能正常工作。")
        else:
            print(f"\n[WARNING] {total - passed} 个测试失败，需要进一步检查。")
        
        return passed == total


def main():
    """主测试函数"""
    tester = ComprehensiveTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    main()
