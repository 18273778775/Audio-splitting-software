#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整功能测试脚本
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import AudioSplitter
from create_test_audio import create_test_audio


def test_progress_callback(progress, message):
    """进度回调函数"""
    print(f"进度: {progress}% - {message}")


def test_audio_splitting():
    """测试音频分割功能"""
    print("=== 音频分割工具完整功能测试 ===\n")
    
    # 1. 创建测试音频文件
    print("1. 创建测试音频文件...")
    test_file = create_test_audio()
    
    if not os.path.exists(test_file):
        print("错误: 测试音频文件创建失败")
        return False
    
    # 2. 创建AudioSplitter实例
    print("\n2. 创建AudioSplitter实例...")
    splitter = AudioSplitter()
    
    # 3. 测试音频分割
    print("\n3. 测试音频分割（每2秒分割一次）...")
    success, message, output_files = splitter.split_audio(
        test_file, 2, test_progress_callback
    )
    
    if success:
        print(f"\n✓ 分割成功: {message}")
        print(f"输出文件数量: {len(output_files)}")
        
        # 检查输出文件
        print("\n输出文件列表:")
        for i, file_path in enumerate(output_files, 1):
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  {i}. {os.path.basename(file_path)} ({file_size} 字节)")
            else:
                print(f"  {i}. {os.path.basename(file_path)} (文件不存在)")
        
        # 检查输出目录
        output_dir = Path(test_file).parent / "output"
        if output_dir.exists():
            print(f"\n输出目录: {output_dir}")
            print(f"目录中的文件数量: {len(list(output_dir.glob('*')))}")
        
        return True
    else:
        print(f"\n✗ 分割失败: {message}")
        return False


def test_error_handling():
    """测试错误处理"""
    print("\n=== 错误处理测试 ===\n")
    
    splitter = AudioSplitter()
    
    # 测试不存在的文件
    print("1. 测试不存在的文件...")
    success, message, _ = splitter.split_audio("nonexistent.wav", 1)
    print(f"结果: {message}")
    
    # 测试不支持的格式
    print("\n2. 测试不支持的格式...")
    success, message, _ = splitter.split_audio("test.txt", 1)
    print(f"结果: {message}")
    
    print("\n错误处理测试完成。")


def cleanup():
    """清理测试文件"""
    print("\n=== 清理测试文件 ===")
    
    # 删除测试音频文件
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"已删除: {test_file}")
    
    # 删除输出目录
    output_dir = Path("output")
    if output_dir.exists():
        for file in output_dir.glob("*"):
            file.unlink()
        output_dir.rmdir()
        print(f"已删除输出目录: {output_dir}")


def main():
    """主测试函数"""
    try:
        # 测试音频分割功能
        success = test_audio_splitting()
        
        # 测试错误处理
        test_error_handling()
        
        if success:
            print("\n🎉 所有测试通过！音频分割工具工作正常。")
        else:
            print("\n❌ 测试失败，请检查代码。")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
    
    finally:
        # 清理测试文件
        cleanup()


if __name__ == "__main__":
    main()
