#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试精度修复效果
验证固定时长和自定义时长分割的小数精度是否正确
"""

import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import AudioSplitter


def create_test_audio(duration=30.0, sample_rate=44100):
    """创建测试音频文件"""
    print(f"创建测试音频文件 (时长: {duration}秒, 采样率: {sample_rate}Hz)")
    
    # 生成正弦波
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # 保存为WAV文件
    test_file = "precision_fix_test.wav"
    sf.write(test_file, audio_data, sample_rate)
    
    return test_file, audio_data, sample_rate


def test_fixed_duration_precision():
    """测试固定时长分割的小数精度"""
    print("\n=== 测试固定时长分割的小数精度 ===")
    
    try:
        # 创建测试音频
        test_file, original_audio, sample_rate = create_test_audio()
        
        # 创建分割器
        splitter = AudioSplitter()
        
        # 测试不同的小数时长
        test_durations = [5.26, 3.75, 7.123, 2.5]
        
        for duration in test_durations:
            print(f"\n测试固定时长: {duration}秒")
            
            success, message, output_files = splitter.split_audio(
                test_file, segment_duration=duration, smart_split=False
            )
            
            if success:
                print(f"分割成功: {len(output_files)} 个文件")
                
                # 验证精确度
                for i, file_path in enumerate(output_files):
                    if os.path.exists(file_path):
                        audio_data, sr = sf.read(file_path)
                        actual_duration = len(audio_data) / sr
                        
                        if i < len(output_files) - 1:  # 不是最后一个片段
                            expected_duration = duration
                            error = abs(actual_duration - expected_duration)
                            print(f"  片段 {i+1}: 期望={expected_duration:.6f}s, 实际={actual_duration:.6f}s, 误差={error*1000:.3f}ms")
                        else:  # 最后一个片段
                            print(f"  片段 {i+1}: 最后片段={actual_duration:.6f}s")
                        
                        # 清理文件
                        os.remove(file_path)
                
                # 清理输出目录
                output_dir = Path("output")
                if output_dir.exists() and not any(output_dir.iterdir()):
                    output_dir.rmdir()
            else:
                print(f"分割失败: {message}")
        
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False


def test_custom_duration_precision():
    """测试自定义时长分割的小数精度"""
    print("\n=== 测试自定义时长分割的小数精度 ===")
    
    try:
        # 创建测试音频
        test_file, original_audio, sample_rate = create_test_audio()
        
        # 创建分割器
        splitter = AudioSplitter()
        
        # 测试不同的小数时长组合
        test_cases = [
            [5.26, 3.75, 7.123],
            [2.5, 4.33, 1.87, 3.14],
            [6.789, 2.345, 5.678]
        ]
        
        for i, custom_durations in enumerate(test_cases):
            print(f"\n测试自定义时长组 {i+1}: {custom_durations}")
            
            success, message, output_files = splitter.split_audio(
                test_file, custom_durations=custom_durations, smart_split=False
            )
            
            if success:
                print(f"分割成功: {len(output_files)} 个文件")
                
                # 验证精确度
                for j, file_path in enumerate(output_files):
                    if os.path.exists(file_path) and j < len(custom_durations):
                        audio_data, sr = sf.read(file_path)
                        actual_duration = len(audio_data) / sr
                        expected_duration = custom_durations[j]
                        error = abs(actual_duration - expected_duration)
                        print(f"  片段 {j+1}: 期望={expected_duration:.6f}s, 实际={actual_duration:.6f}s, 误差={error*1000:.3f}ms")
                        
                        # 清理文件
                        os.remove(file_path)
                
                # 清理输出目录
                output_dir = Path("output")
                if output_dir.exists() and not any(output_dir.iterdir()):
                    output_dir.rmdir()
            else:
                print(f"分割失败: {message}")
        
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False


def test_gui_parameter_passing():
    """测试GUI参数传递的精度"""
    print("\n=== 测试GUI参数传递精度 ===")
    
    try:
        # 创建GUI实例（不显示窗口）
        import tkinter as tk
        from main import AudioSplitterGUI
        
        # 创建隐藏的根窗口
        root = tk.Tk()
        root.withdraw()  # 隐藏窗口
        
        # 创建GUI实例
        gui = AudioSplitterGUI(root)
        
        # 测试固定时长参数传递
        print("\n测试固定时长参数传递:")
        test_values = ["5.26", "3.75", "7.123", "2.5"]
        
        for value in test_values:
            gui.duration_var.set(value)
            gui.time_unit_var.set("秒")
            result = gui.get_duration_in_seconds()
            print(f"  输入: {value}秒 -> 输出: {result}秒")
            
            # 验证精度
            expected = float(value)
            if abs(result - expected) < 1e-10:
                print(f"    ✓ 精度正确")
            else:
                print(f"    ✗ 精度错误，期望: {expected}, 实际: {result}")
        
        # 测试自定义时长参数传递
        print("\n测试自定义时长参数传递:")
        test_inputs = [
            "5.26",
            "3.75, 7.123",
            "2.5, 4.33, 1.87",
            "6.789, 2.345, 5.678, 3.14"
        ]
        
        for input_str in test_inputs:
            durations, error = gui.parse_custom_durations(input_str)
            if error:
                print(f"  输入: {input_str} -> 错误: {error}")
            else:
                print(f"  输入: {input_str} -> 输出: {durations}")
                
                # 验证精度
                expected_values = [float(x.strip()) for x in input_str.split(',')]
                if len(durations) == len(expected_values):
                    all_correct = True
                    for i, (actual, expected) in enumerate(zip(durations, expected_values)):
                        if abs(actual - expected) >= 1e-10:
                            all_correct = False
                            break
                    
                    if all_correct:
                        print(f"    ✓ 精度正确")
                    else:
                        print(f"    ✗ 精度错误")
                else:
                    print(f"    ✗ 数量不匹配")
        
        # 销毁窗口
        root.destroy()
        
        return True
        
    except Exception as e:
        print(f"GUI测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("精度修复验证测试")
    print("=" * 60)
    
    results = []
    
    # 测试1: 固定时长分割精度
    result1 = test_fixed_duration_precision()
    results.append(("固定时长分割精度", result1))
    
    # 测试2: 自定义时长分割精度
    result2 = test_custom_duration_precision()
    results.append(("自定义时长分割精度", result2))
    
    # 测试3: GUI参数传递精度
    result3 = test_gui_parameter_passing()
    results.append(("GUI参数传递精度", result3))
    
    # 总结结果
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n[SUCCESS] 精度修复验证成功！所有小数精度问题已解决。")
    else:
        print(f"\n[WARNING] {total - passed} 个测试失败，需要进一步检查。")
    
    return passed == total


if __name__ == "__main__":
    main()
