#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频分割精确度验证测试脚本
验证所有分割模式的精确度是否达到毫秒级（小数点后三位）
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
from create_test_audio import create_test_audio


class PrecisionTester:
    """精确度测试类"""
    
    def __init__(self):
        self.splitter = AudioSplitter()
        self.test_results = []
        
    def create_test_audio_file(self, duration=30.0, sample_rate=44100):
        """创建测试音频文件"""
        print(f"创建测试音频文件 (时长: {duration}秒, 采样率: {sample_rate}Hz)...")
        
        # 生成测试音频数据（正弦波）
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        frequency = 440  # A4音符
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # 保存为WAV文件
        test_file = "precision_test_audio.wav"
        sf.write(test_file, audio_data, sample_rate)
        
        return test_file, audio_data, sample_rate
    
    def measure_split_precision(self, output_files, expected_durations, sample_rate):
        """测量分割精确度"""
        actual_durations = []
        precision_errors = []
        
        for i, file_path in enumerate(output_files):
            if os.path.exists(file_path):
                # 加载音频文件并测量实际时长
                audio_data, sr = librosa.load(file_path, sr=None)
                actual_duration = len(audio_data) / sr
                actual_durations.append(actual_duration)
                
                # 计算精确度误差
                expected_duration = expected_durations[i] if i < len(expected_durations) else expected_durations[-1]
                error = abs(actual_duration - expected_duration)
                precision_errors.append(error)
                
                print(f"  片段 {i+1}: 期望={expected_duration:.6f}s, 实际={actual_duration:.6f}s, 误差={error:.6f}s ({error*1000:.3f}ms)")
            else:
                print(f"  片段 {i+1}: 文件不存在")
                actual_durations.append(0)
                precision_errors.append(float('inf'))
        
        return actual_durations, precision_errors
    
    def test_continuity(self, output_files, original_audio, sample_rate):
        """测试音频连续性"""
        print("  测试音频连续性...")
        
        # 重新拼接分割后的音频
        reconstructed_audio = []
        
        for file_path in output_files:
            if os.path.exists(file_path):
                audio_data, sr = librosa.load(file_path, sr=None)
                reconstructed_audio.extend(audio_data)
        
        reconstructed_audio = np.array(reconstructed_audio)
        
        # 比较原始音频和重构音频
        min_length = min(len(original_audio), len(reconstructed_audio))
        original_trimmed = original_audio[:min_length]
        reconstructed_trimmed = reconstructed_audio[:min_length]
        
        # 计算差异
        difference = np.abs(original_trimmed - reconstructed_trimmed)
        max_difference = np.max(difference)
        mean_difference = np.mean(difference)
        
        print(f"    原始音频长度: {len(original_audio)} 采样点")
        print(f"    重构音频长度: {len(reconstructed_audio)} 采样点")
        print(f"    长度差异: {abs(len(original_audio) - len(reconstructed_audio))} 采样点")
        print(f"    最大幅度差异: {max_difference:.8f}")
        print(f"    平均幅度差异: {mean_difference:.8f}")
        
        # 判断连续性是否良好（允许极小的数值误差）
        length_diff = abs(len(original_audio) - len(reconstructed_audio))
        is_continuous = length_diff <= 1 and max_difference < 1e-6
        
        return is_continuous, length_diff, max_difference
    
    def test_fixed_duration_precision(self, test_file, original_audio, sample_rate):
        """测试固定时长分割精确度"""
        print("\n=== 测试固定时长分割精确度 ===")
        
        test_durations = [3.0, 5.5, 7.25]  # 包含非整数时长
        
        for duration in test_durations:
            print(f"\n测试固定时长: {duration}秒")
            
            # 非智能分割
            success, message, output_files = self.splitter.split_audio(
                test_file, segment_duration=duration, smart_split=False
            )
            
            if success:
                # 计算期望的分割时长
                total_duration = len(original_audio) / sample_rate
                num_segments = int(np.ceil(total_duration / duration))
                expected_durations = [duration] * (num_segments - 1)
                last_duration = total_duration - (num_segments - 1) * duration
                expected_durations.append(last_duration)
                
                # 测量精确度
                actual_durations, errors = self.measure_split_precision(output_files, expected_durations, sample_rate)
                
                # 测试连续性
                is_continuous, length_diff, max_diff = self.test_continuity(output_files, original_audio, sample_rate)
                
                # 记录结果
                max_error_ms = max(errors) * 1000 if errors else float('inf')
                result = {
                    'mode': 'fixed',
                    'duration': duration,
                    'smart_split': False,
                    'max_error_ms': max_error_ms,
                    'is_continuous': is_continuous,
                    'length_diff': length_diff,
                    'max_amplitude_diff': max_diff
                }
                self.test_results.append(result)
                
                print(f"  最大误差: {max_error_ms:.3f}ms")
                print(f"  连续性: {'通过' if is_continuous else '失败'}")
                
                # 清理输出文件
                self.cleanup_output_files(output_files)
            else:
                print(f"  分割失败: {message}")
    
    def test_custom_duration_precision(self, test_file, original_audio, sample_rate):
        """测试自定义时长分割精确度"""
        print("\n=== 测试自定义时长分割精确度 ===")
        
        test_cases = [
            [2.5, 3.7, 4.2, 5.1],  # 非整数时长
            [1.0, 2.0, 3.0, 4.0],  # 整数时长
            [0.5, 1.5, 2.5, 3.5]   # 半秒时长
        ]
        
        for i, custom_durations in enumerate(test_cases):
            print(f"\n测试自定义时长组 {i+1}: {custom_durations}")
            
            # 非智能分割
            success, message, output_files = self.splitter.split_audio(
                test_file, custom_durations=custom_durations, smart_split=False
            )
            
            if success:
                # 期望时长就是自定义时长
                expected_durations = custom_durations.copy()
                
                # 测量精确度
                actual_durations, errors = self.measure_split_precision(output_files, expected_durations, sample_rate)
                
                # 测试连续性（只测试指定的片段）
                specified_files = output_files[:len(custom_durations)]
                is_continuous, length_diff, max_diff = self.test_continuity(specified_files, original_audio, sample_rate)
                
                # 记录结果
                max_error_ms = max(errors[:len(custom_durations)]) * 1000 if errors else float('inf')
                result = {
                    'mode': 'custom',
                    'durations': custom_durations,
                    'smart_split': False,
                    'max_error_ms': max_error_ms,
                    'is_continuous': is_continuous,
                    'length_diff': length_diff,
                    'max_amplitude_diff': max_diff
                }
                self.test_results.append(result)
                
                print(f"  最大误差: {max_error_ms:.3f}ms")
                print(f"  连续性: {'通过' if is_continuous else '失败'}")
                
                # 清理输出文件
                self.cleanup_output_files(output_files)
            else:
                print(f"  分割失败: {message}")
    
    def test_smart_split_precision(self, test_file, original_audio, sample_rate):
        """测试智能分割精确度"""
        print("\n=== 测试智能分割精确度 ===")
        
        test_durations = [4.0, 6.5]  # 智能分割测试时长
        
        for duration in test_durations:
            print(f"\n测试智能分割: {duration}秒")
            
            # 智能分割
            success, message, output_files = self.splitter.split_audio(
                test_file, segment_duration=duration, smart_split=True, search_range=1.0
            )
            
            if success:
                # 对于智能分割，我们主要测试连续性而不是精确的时长匹配
                is_continuous, length_diff, max_diff = self.test_continuity(output_files, original_audio, sample_rate)
                
                # 记录结果
                result = {
                    'mode': 'smart',
                    'duration': duration,
                    'smart_split': True,
                    'is_continuous': is_continuous,
                    'length_diff': length_diff,
                    'max_amplitude_diff': max_diff
                }
                self.test_results.append(result)
                
                print(f"  连续性: {'通过' if is_continuous else '失败'}")
                print(f"  长度差异: {length_diff} 采样点")
                
                # 清理输出文件
                self.cleanup_output_files(output_files)
            else:
                print(f"  分割失败: {message}")
    
    def cleanup_output_files(self, output_files):
        """清理输出文件"""
        for file_path in output_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # 删除输出目录（如果为空）
        output_dir = Path("output")
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
    
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "="*60)
        print("精确度测试报告")
        print("="*60)
        
        # 统计结果
        total_tests = len(self.test_results)
        passed_precision = 0
        passed_continuity = 0
        
        print(f"\n总测试数量: {total_tests}")
        print("\n详细结果:")
        
        for i, result in enumerate(self.test_results, 1):
            mode = result['mode']
            smart = "智能" if result.get('smart_split', False) else "非智能"
            
            print(f"\n{i}. {mode.upper()}模式 ({smart})")
            
            if 'duration' in result:
                print(f"   时长: {result['duration']}秒")
            elif 'durations' in result:
                print(f"   时长: {result['durations']}")
            
            if 'max_error_ms' in result:
                error_ms = result['max_error_ms']
                precision_ok = error_ms <= 1.0  # 1毫秒以内认为合格
                print(f"   最大误差: {error_ms:.3f}ms {'通过' if precision_ok else '失败'}")
                if precision_ok:
                    passed_precision += 1

            continuity_ok = result['is_continuous']
            print(f"   连续性: {'通过' if continuity_ok else '失败'}")
            if continuity_ok:
                passed_continuity += 1
            
            print(f"   长度差异: {result['length_diff']} 采样点")
            print(f"   幅度差异: {result['max_amplitude_diff']:.8f}")
        
        # 总结
        print(f"\n总结:")
        print(f"精确度测试通过: {passed_precision}/{total_tests}")
        print(f"连续性测试通过: {passed_continuity}/{total_tests}")
        
        # 判断是否达到要求
        precision_rate = passed_precision / total_tests if total_tests > 0 else 0
        continuity_rate = passed_continuity / total_tests if total_tests > 0 else 0
        
        print(f"\n第一阶段要求验证:")
        print(f"[OK] 精确度达到毫秒级: {'是' if precision_rate >= 0.9 else '否'} ({precision_rate*100:.1f}%)")
        print(f"[OK] 音频完美连续性: {'是' if continuity_rate >= 0.9 else '否'} ({continuity_rate*100:.1f}%)")
        print(f"[OK] 所有模式精确度统一: {'是' if precision_rate >= 0.9 and continuity_rate >= 0.9 else '否'}")
        
        return precision_rate >= 0.9 and continuity_rate >= 0.9


def main():
    """主测试函数"""
    print("音频分割精确度验证测试")
    print("="*60)
    
    tester = PrecisionTester()
    
    try:
        # 创建测试音频文件
        test_file, original_audio, sample_rate = tester.create_test_audio_file()
        
        # 执行各种精确度测试
        tester.test_fixed_duration_precision(test_file, original_audio, sample_rate)
        tester.test_custom_duration_precision(test_file, original_audio, sample_rate)
        tester.test_smart_split_precision(test_file, original_audio, sample_rate)
        
        # 生成报告
        all_passed = tester.generate_report()
        
        if all_passed:
            print("\n[SUCCESS] 第一阶段验证通过！所有分割模式都达到了毫秒级精确度要求。")
        else:
            print("\n[FAILED] 第一阶段验证未通过，需要进一步优化精确度实现。")
        
        return all_passed
        
    except Exception as e:
        print(f"\n[ERROR] 测试过程中发生错误: {e}")
        return False
    
    finally:
        # 清理测试文件
        test_file = "precision_test_audio.wav"
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n已清理测试文件: {test_file}")


if __name__ == "__main__":
    main()
