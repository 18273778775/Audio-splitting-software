#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证精度修复的简单脚本
"""

import os
import sys
import numpy as np
import soundfile as sf

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import AudioSplitter


def quick_precision_test():
    """快速精度验证测试"""
    print("快速精度验证测试")
    print("=" * 40)
    
    # 创建简单测试音频
    duration = 20.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    test_file = "quick_test.wav"
    sf.write(test_file, audio_data, sample_rate)
    
    splitter = AudioSplitter()
    
    # 测试用户报告的问题：5.26秒
    print("\n测试用户报告的问题：5.26秒固定时长分割")
    success, message, output_files = splitter.split_audio(
        test_file, segment_duration=5.26, smart_split=False
    )
    
    if success:
        print(f"分割成功，生成 {len(output_files)} 个文件")
        
        for i, file_path in enumerate(output_files):
            if os.path.exists(file_path):
                audio_data, sr = sf.read(file_path)
                actual_duration = len(audio_data) / sr
                
                if i < len(output_files) - 1:  # 不是最后一个片段
                    print(f"片段 {i+1}: {actual_duration:.6f}秒 (期望: 5.260000秒)")
                    if abs(actual_duration - 5.26) < 0.001:
                        print("  ✓ 精度正确！")
                    else:
                        print("  ✗ 精度错误！")
                else:
                    print(f"片段 {i+1}: {actual_duration:.6f}秒 (最后片段)")
                
                # 清理文件
                os.remove(file_path)
        
        # 清理输出目录
        if os.path.exists("output") and not os.listdir("output"):
            os.rmdir("output")
    else:
        print(f"分割失败: {message}")
    
    # 测试自定义时长：5.26
    print("\n测试自定义时长分割：5.26秒")
    success, message, output_files = splitter.split_audio(
        test_file, custom_durations=[5.26], smart_split=False
    )
    
    if success:
        print(f"分割成功，生成 {len(output_files)} 个文件")
        
        if output_files and os.path.exists(output_files[0]):
            audio_data, sr = sf.read(output_files[0])
            actual_duration = len(audio_data) / sr
            print(f"片段 1: {actual_duration:.6f}秒 (期望: 5.260000秒)")
            
            if abs(actual_duration - 5.26) < 0.001:
                print("  ✓ 精度正确！")
            else:
                print("  ✗ 精度错误！")
            
            # 清理文件
            for file_path in output_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # 清理输出目录
            if os.path.exists("output") and not os.listdir("output"):
                os.rmdir("output")
    else:
        print(f"分割失败: {message}")
    
    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("\n验证完成！")


if __name__ == "__main__":
    quick_precision_test()
