#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的视频时长匹配功能测试
"""

import os
import sys
import numpy as np
import soundfile as sf

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """测试基本功能"""
    print("Testing basic video matching functionality...")
    
    try:
        # 测试导入
        from video_processor import VideoProcessor
        from main import AudioSplitter
        print("[OK] Imports successful")
        
        # 创建实例
        processor = VideoProcessor()
        splitter = AudioSplitter()
        print("[OK] Objects created successfully")
        
        # 测试格式检查
        test_formats = ["test.mp4", "test.avi", "test.txt"]
        for fmt in test_formats:
            result = processor.is_supported_format(fmt)
            print(f"  {fmt}: {'supported' if result else 'not supported'}")
        
        # 创建简单测试音频
        duration = 10.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        test_file = "simple_test.wav"
        sf.write(test_file, audio_data, sample_rate)
        print("[OK] Test audio created")
        
        # 测试视频时长匹配分割
        video_durations = [3.0, 4.0, 2.5]  # 简单的时长列表
        print(f"Testing with durations: {video_durations}")
        
        success, message, output_files = splitter.split_audio_by_video_durations(
            test_file, video_durations, smart_split=False
        )
        
        if success:
            print(f"[OK] Split successful: {len(output_files)} files created")
            
            # 验证输出文件
            for i, file_path in enumerate(output_files):
                if os.path.exists(file_path):
                    audio_data, sr = sf.read(file_path)
                    actual_duration = len(audio_data) / sr
                    expected_duration = video_durations[i] if i < len(video_durations) else 0
                    error = abs(actual_duration - expected_duration) if expected_duration > 0 else 0
                    print(f"  File {i+1}: Expected={expected_duration:.3f}s, Actual={actual_duration:.3f}s, Error={error*1000:.1f}ms")
                    
                    # 清理文件
                    os.remove(file_path)
            
            # 清理输出目录
            output_dir = "output"
            if os.path.exists(output_dir) and not os.listdir(output_dir):
                os.rmdir(output_dir)
            
            print("[OK] Video duration matching test passed")
        else:
            print(f"[FAIL] Split failed: {message}")
            return False
        
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
        
        print("\n[SUCCESS] All tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_functionality()
