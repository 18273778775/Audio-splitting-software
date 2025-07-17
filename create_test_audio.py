#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建测试音频文件
"""

import numpy as np
import soundfile as sf
import os


def create_test_audio():
    """创建一个简单的测试音频文件"""
    # 生成5秒的正弦波音频（440Hz，A音）
    duration = 5  # 秒
    sample_rate = 44100  # 采样率
    frequency = 440  # 频率（Hz）
    
    # 生成时间轴
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 生成正弦波
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # 保存为WAV文件
    output_file = "test_audio.wav"
    sf.write(output_file, audio_data, sample_rate)
    
    print(f"测试音频文件已创建: {output_file}")
    print(f"时长: {duration}秒")
    print(f"采样率: {sample_rate}Hz")
    print(f"文件大小: {os.path.getsize(output_file)} 字节")
    
    return output_file


if __name__ == "__main__":
    create_test_audio()
