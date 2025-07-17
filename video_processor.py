#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理模块
用于读取视频文件时长和相关信息
"""

import os
import sys
from pathlib import Path

try:
    import cv2
except ImportError as e:
    print(f"请安装视频处理库:")
    print(f"pip install opencv-python")
    print(f"错误详情: {e}")
    sys.exit(1)


class VideoProcessor:
    """视频处理类"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    
    def is_supported_format(self, file_path):
        """检查视频文件格式是否支持"""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def get_video_duration(self, file_path):
        """
        获取视频文件时长
        
        Args:
            file_path (str): 视频文件路径
            
        Returns:
            tuple: (success, duration_seconds, error_message)
        """
        try:
            if not os.path.exists(file_path):
                return False, 0.0, f"文件不存在: {file_path}"
            
            if not self.is_supported_format(file_path):
                return False, 0.0, f"不支持的视频格式: {Path(file_path).suffix}"
            
            # 使用OpenCV读取视频信息
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                return False, 0.0, f"无法打开视频文件: {file_path}"
            
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
            
            cap.release()
            
            if fps <= 0:
                return False, 0.0, f"无法获取视频帧率: {file_path}"
            
            # 计算时长（秒）
            duration = frame_count / fps
            
            return True, duration, ""
            
        except Exception as e:
            return False, 0.0, f"读取视频时长时发生错误: {str(e)}"
    
    def get_video_info(self, file_path):
        """
        获取视频文件详细信息
        
        Args:
            file_path (str): 视频文件路径
            
        Returns:
            dict: 视频信息字典
        """
        try:
            if not os.path.exists(file_path):
                return {"error": f"文件不存在: {file_path}"}
            
            if not self.is_supported_format(file_path):
                return {"error": f"不支持的视频格式: {Path(file_path).suffix}"}
            
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                return {"error": f"无法打开视频文件: {file_path}"}
            
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            if fps <= 0:
                return {"error": f"无法获取视频帧率: {file_path}"}
            
            duration = frame_count / fps
            file_size = os.path.getsize(file_path)
            
            return {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "file_size": file_size,
                "format": Path(file_path).suffix.lower()
            }
            
        except Exception as e:
            return {"error": f"读取视频信息时发生错误: {str(e)}"}
    
    def batch_get_video_durations(self, file_paths, progress_callback=None):
        """
        批量获取多个视频文件的时长
        
        Args:
            file_paths (list): 视频文件路径列表
            progress_callback (callable): 进度回调函数
            
        Returns:
            list: 视频信息列表，每个元素包含 {file_path, duration, error}
        """
        results = []
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            success, duration, error = self.get_video_duration(file_path)
            
            result = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "duration": duration if success else 0.0,
                "success": success,
                "error": error if not success else ""
            }
            
            results.append(result)
            
            # 更新进度
            if progress_callback:
                progress = int((i + 1) / total_files * 100)
                progress_callback(progress, f"正在读取视频 {i+1}/{total_files}: {os.path.basename(file_path)}")
        
        return results
    
    def validate_video_files(self, file_paths):
        """
        验证视频文件列表
        
        Args:
            file_paths (list): 视频文件路径列表
            
        Returns:
            tuple: (valid_files, invalid_files, error_messages)
        """
        valid_files = []
        invalid_files = []
        error_messages = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                invalid_files.append(file_path)
                error_messages.append(f"文件不存在: {os.path.basename(file_path)}")
                continue
            
            if not self.is_supported_format(file_path):
                invalid_files.append(file_path)
                error_messages.append(f"不支持的格式: {os.path.basename(file_path)}")
                continue
            
            # 尝试打开文件验证
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    
                    if fps > 0:
                        valid_files.append(file_path)
                    else:
                        invalid_files.append(file_path)
                        error_messages.append(f"无效的视频文件: {os.path.basename(file_path)}")
                else:
                    invalid_files.append(file_path)
                    error_messages.append(f"无法打开视频文件: {os.path.basename(file_path)}")
            except Exception as e:
                invalid_files.append(file_path)
                error_messages.append(f"验证失败: {os.path.basename(file_path)} - {str(e)}")
        
        return valid_files, invalid_files, error_messages
    
    def format_duration(self, duration_seconds):
        """
        格式化时长显示
        
        Args:
            duration_seconds (float): 时长（秒）
            
        Returns:
            str: 格式化的时长字符串
        """
        if duration_seconds < 0:
            return "00:00.000"
        
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = duration_seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        else:
            return f"{minutes:02d}:{seconds:06.3f}"
    
    def get_supported_formats_string(self):
        """获取支持的视频格式字符串"""
        return "支持的视频格式: " + ", ".join(self.supported_formats)


# 测试函数
def test_video_processor():
    """测试视频处理功能"""
    processor = VideoProcessor()
    
    print("视频处理器测试")
    print("=" * 40)
    print(processor.get_supported_formats_string())
    
    # 测试格式检查
    test_files = [
        "test.mp4",
        "test.avi", 
        "test.mov",
        "test.txt",
        "test.mp3"
    ]
    
    print("\n格式检查测试:")
    for file in test_files:
        is_supported = processor.is_supported_format(file)
        print(f"  {file}: {'支持' if is_supported else '不支持'}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_video_processor()
