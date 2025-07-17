#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频时长匹配功能测试脚本
"""

import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_processor import VideoProcessor
from main import AudioSplitter


def create_test_video_info():
    """创建测试视频信息（模拟）"""
    # 模拟几个视频文件的时长
    test_videos = [
        {"name": "video1.mp4", "duration": 5.123},
        {"name": "video2.mp4", "duration": 7.456},
        {"name": "video3.mp4", "duration": 3.789},
        {"name": "video4.mp4", "duration": 6.234}
    ]
    
    return test_videos


def create_test_audio(duration=25.0, sample_rate=44100):
    """创建测试音频文件"""
    print(f"创建测试音频文件 (时长: {duration}秒, 采样率: {sample_rate}Hz)...")
    
    # 生成测试音频数据（正弦波）
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    frequency = 440  # A4音符
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # 保存为WAV文件
    test_file = "test_audio_for_video_matching.wav"
    sf.write(test_file, audio_data, sample_rate)
    
    return test_file, audio_data, sample_rate


def test_video_processor():
    """测试视频处理器"""
    print("测试视频处理器...")
    print("=" * 50)
    
    processor = VideoProcessor()
    
    # 测试支持的格式
    print("支持的视频格式:")
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
    
    print("\n视频处理器测试完成！")
    return True


def test_video_duration_matching():
    """测试视频时长匹配分割功能"""
    print("\n测试视频时长匹配分割功能...")
    print("=" * 50)
    
    try:
        # 创建测试音频
        test_file, original_audio, sample_rate = create_test_audio()
        
        # 创建音频分割器
        splitter = AudioSplitter()
        
        # 模拟视频时长列表
        test_videos = create_test_video_info()
        video_durations = [video["duration"] for video in test_videos]
        
        print(f"测试视频时长列表: {video_durations}")
        print(f"总时长: {sum(video_durations):.3f}秒")
        
        # 执行视频时长匹配分割
        print("\n执行视频时长匹配分割...")
        success, message, output_files = splitter.split_audio_by_video_durations(
            test_file, video_durations, smart_split=False
        )
        
        if success:
            print(f"✓ 分割成功: {message}")
            print(f"生成文件数量: {len(output_files)}")
            
            # 验证分割精确度
            print("\n验证分割精确度:")
            total_error = 0
            
            for i, (file_path, expected_duration) in enumerate(zip(output_files, video_durations)):
                if os.path.exists(file_path):
                    # 读取分割后的音频文件
                    audio_data, sr = sf.read(file_path)
                    actual_duration = len(audio_data) / sr
                    error = abs(actual_duration - expected_duration)
                    total_error += error
                    
                    print(f"  片段 {i+1} ({test_videos[i]['name']}): "
                          f"期望={expected_duration:.6f}s, "
                          f"实际={actual_duration:.6f}s, "
                          f"误差={error:.6f}s ({error*1000:.3f}ms)")
                else:
                    print(f"  片段 {i+1}: 文件不存在")
            
            avg_error_ms = (total_error / len(video_durations)) * 1000
            print(f"\n平均误差: {avg_error_ms:.3f}ms")
            
            # 清理输出文件
            for file_path in output_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # 删除输出目录
            output_dir = Path("output")
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()
            
            # 判断精确度
            if avg_error_ms <= 1.0:
                print("✓ 精确度测试通过 (误差 <= 1ms)")
                return True
            else:
                print("✗ 精确度测试失败 (误差 > 1ms)")
                return False
        else:
            print(f"✗ 分割失败: {message}")
            return False
            
    except Exception as e:
        print(f"✗ 测试过程中发生错误: {e}")
        return False
    
    finally:
        # 清理测试文件
        test_file = "test_audio_for_video_matching.wav"
        if os.path.exists(test_file):
            os.remove(test_file)


def test_smart_video_matching():
    """测试智能视频时长匹配"""
    print("\n测试智能视频时长匹配...")
    print("=" * 50)
    
    try:
        # 创建测试音频
        test_file, original_audio, sample_rate = create_test_audio()
        
        # 创建音频分割器
        splitter = AudioSplitter()
        
        # 模拟视频时长列表
        test_videos = create_test_video_info()
        video_durations = [video["duration"] for video in test_videos]
        
        print(f"测试视频时长列表: {video_durations}")
        
        # 执行智能视频时长匹配分割
        print("\n执行智能视频时长匹配分割...")
        success, message, output_files = splitter.split_audio_by_video_durations(
            test_file, video_durations, smart_split=True, search_range=1.0
        )
        
        if success:
            print(f"✓ 智能分割成功: {message}")
            print(f"生成文件数量: {len(output_files)}")
            
            # 验证文件连续性
            print("\n验证音频连续性:")
            reconstructed_audio = []
            
            for i, file_path in enumerate(output_files):
                if os.path.exists(file_path):
                    audio_data, sr = sf.read(file_path)
                    reconstructed_audio.extend(audio_data)
                    actual_duration = len(audio_data) / sr
                    print(f"  片段 {i+1}: 时长={actual_duration:.6f}s")
            
            # 比较原始音频和重构音频
            reconstructed_audio = np.array(reconstructed_audio)
            min_length = min(len(original_audio), len(reconstructed_audio))
            
            length_diff = abs(len(original_audio) - len(reconstructed_audio))
            print(f"\n连续性检查:")
            print(f"  原始音频长度: {len(original_audio)} 采样点")
            print(f"  重构音频长度: {len(reconstructed_audio)} 采样点")
            print(f"  长度差异: {length_diff} 采样点")
            
            # 清理输出文件
            for file_path in output_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # 删除输出目录
            output_dir = Path("output")
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()
            
            if length_diff <= 1:
                print("✓ 连续性测试通过")
                return True
            else:
                print("✗ 连续性测试失败")
                return False
        else:
            print(f"✗ 智能分割失败: {message}")
            return False
            
    except Exception as e:
        print(f"✗ 测试过程中发生错误: {e}")
        return False
    
    finally:
        # 清理测试文件
        test_file = "test_audio_for_video_matching.wav"
        if os.path.exists(test_file):
            os.remove(test_file)


def main():
    """主测试函数"""
    print("视频时长匹配功能测试")
    print("=" * 60)
    
    test_results = []
    
    # 测试1: 视频处理器
    result1 = test_video_processor()
    test_results.append(("视频处理器", result1))
    
    # 测试2: 视频时长匹配分割
    result2 = test_video_duration_matching()
    test_results.append(("视频时长匹配分割", result2))
    
    # 测试3: 智能视频时长匹配
    result3 = test_smart_video_matching()
    test_results.append(("智能视频时长匹配", result3))
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 第二阶段视频时长匹配功能测试全部通过！")
        return True
    else:
        print("\n❌ 部分测试失败，需要进一步调试。")
        return False


if __name__ == "__main__":
    main()
