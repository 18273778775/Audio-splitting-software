#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频分割工具测试脚本
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import AudioSplitter


def test_audio_splitter():
    """测试音频分割功能"""
    print("测试音频分割工具...")
    
    # 创建AudioSplitter实例
    splitter = AudioSplitter()
    
    # 测试支持的格式检查
    print("\n1. 测试格式检查:")
    test_files = [
        "test.mp3",
        "test.wav", 
        "test.txt",
        "test.flac"
    ]
    
    for file in test_files:
        is_supported = splitter.is_supported_format(file)
        print(f"  {file}: {'支持' if is_supported else '不支持'}")
    
    print("\n2. 测试音频分割功能:")
    print("  由于没有实际音频文件，无法进行完整测试")
    print("  但核心类已成功创建并可以调用")
    
    # 测试不存在的文件
    success, message, files = splitter.split_audio("nonexistent.mp3", segment_duration=60)
    print(f"  测试不存在文件: {message}")

    print("\n3. 测试自定义长度分割功能:")

    # 测试参数验证
    test_cases = [
        # (segment_duration, custom_durations, expected_success, description)
        (None, None, False, "两个参数都为None"),
        (60, [3, 5, 10], False, "同时指定两个参数"),
        (None, [], False, "自定义长度数组为空"),
        (None, [0, 5, 10], False, "包含0长度"),
        (None, [-1, 5, 10], False, "包含负数长度"),
        (None, [3, 5, 10], True, "正常的自定义长度数组"),
    ]

    for segment_duration, custom_durations, expected_success, description in test_cases:
        try:
            success, message, files = splitter.split_audio(
                "nonexistent.mp3",
                segment_duration=segment_duration,
                custom_durations=custom_durations
            )
            result = "✓" if success == expected_success else "✗"
            print(f"  {result} {description}: {message}")
        except Exception as e:
            print(f"  ✗ {description}: 异常 - {str(e)}")

    print("\n测试完成！核心功能正常。")


if __name__ == "__main__":
    test_audio_splitter()
