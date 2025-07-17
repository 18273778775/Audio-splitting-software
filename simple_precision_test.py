#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版精确度测试
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import AudioSplitter


def create_test_audio(duration=20.0, sample_rate=44100):
    """创建测试音频"""
    print(f"Creating test audio: {duration}s at {sample_rate}Hz")
    
    # 生成正弦波
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # 保存文件
    test_file = "simple_test.wav"
    sf.write(test_file, audio_data, sample_rate)
    
    return test_file, audio_data, sample_rate


def test_precision():
    """测试精确度"""
    print("Audio Splitting Precision Test")
    print("=" * 50)
    
    try:
        # 创建测试音频
        test_file, original_audio, sample_rate = create_test_audio()
        
        # 创建分割器
        splitter = AudioSplitter()
        
        # 测试固定时长分割
        print("\nTesting fixed duration split (3.0s)...")
        success, message, output_files = splitter.split_audio(
            test_file, segment_duration=3.0, smart_split=False
        )
        
        if success:
            print(f"Split successful: {len(output_files)} files created")
            
            # 测量精确度
            total_error = 0
            for i, file_path in enumerate(output_files):
                if os.path.exists(file_path):
                    audio_data, sr = librosa.load(file_path, sr=None)
                    actual_duration = len(audio_data) / sr
                    expected_duration = 3.0 if i < len(output_files) - 1 else (20.0 - 3.0 * (len(output_files) - 1))
                    error = abs(actual_duration - expected_duration)
                    total_error += error
                    print(f"  Segment {i+1}: Expected={expected_duration:.6f}s, Actual={actual_duration:.6f}s, Error={error:.6f}s ({error*1000:.3f}ms)")
            
            avg_error_ms = (total_error / len(output_files)) * 1000
            print(f"\nAverage error: {avg_error_ms:.3f}ms")
            
            # 清理文件
            for file_path in output_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # 删除输出目录
            output_dir = Path("output")
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()
            
            # 判断结果
            if avg_error_ms <= 1.0:
                print("RESULT: PRECISION TEST PASSED (Error <= 1ms)")
                return True
            else:
                print("RESULT: PRECISION TEST FAILED (Error > 1ms)")
                return False
        else:
            print(f"Split failed: {message}")
            return False
            
    except Exception as e:
        print(f"Test error: {e}")
        return False
    
    finally:
        # 清理测试文件
        if os.path.exists("simple_test.wav"):
            os.remove("simple_test.wav")


if __name__ == "__main__":
    result = test_precision()
    if result:
        print("\nStage 1 verification: SUCCESS")
    else:
        print("\nStage 1 verification: FAILED")
