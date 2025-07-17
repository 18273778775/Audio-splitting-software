#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建exe文件的脚本
"""

import os
import subprocess
import sys
from pathlib import Path


def build_exe():
    """使用PyInstaller构建exe文件"""
    print("开始构建exe文件...")
    
    # PyInstaller命令参数
    cmd = [
        "pyinstaller",
        "--onefile",  # 打包成单个exe文件
        "--windowed",  # 不显示控制台窗口
        "--name=音频分割工具",  # 设置exe文件名
        "--icon=icon.ico",  # 图标文件（如果存在）
        "--add-data=README.md;.",  # 添加README文件
        "--add-data=使用说明_v3.2_视频时长匹配版.txt;.",  # 添加使用说明
        "--hidden-import=cv2",  # 确保包含OpenCV
        "--hidden-import=video_processor",  # 确保包含视频处理模块
        "main.py"  # 主程序文件
    ]
    
    # 如果没有图标文件，移除图标参数
    if not os.path.exists("icon.ico"):
        cmd = [arg for arg in cmd if not arg.startswith("--icon")]
    
    try:
        # 执行PyInstaller命令
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("构建成功！")
        print(result.stdout)
        
        # 检查生成的exe文件
        exe_path = Path("dist") / "音频分割工具.exe"
        if exe_path.exists():
            file_size = exe_path.stat().st_size / (1024 * 1024)  # MB
            print(f"\n生成的exe文件:")
            print(f"路径: {exe_path}")
            print(f"大小: {file_size:.1f} MB")
            
            return True
        else:
            print("错误: 未找到生成的exe文件")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"构建失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"构建过程中发生错误: {e}")
        return False


def clean_build_files():
    """清理构建过程中生成的临时文件"""
    print("\n清理构建文件...")
    
    import shutil
    
    # 删除build目录
    if os.path.exists("build"):
        shutil.rmtree("build")
        print("已删除build目录")
    
    # 删除spec文件
    spec_file = "音频分割工具.spec"
    if os.path.exists(spec_file):
        os.remove(spec_file)
        print(f"已删除{spec_file}")


def main():
    """主函数"""
    print("=== 音频分割工具打包程序 ===\n")
    
    # 检查依赖
    try:
        import PyInstaller
        print(f"PyInstaller版本: {PyInstaller.__version__}")
    except ImportError:
        print("错误: 未安装PyInstaller")
        print("请运行: pip install pyinstaller")
        return
    
    # 构建exe文件
    success = build_exe()
    
    if success:
        print("\n🎉 打包完成！")
        print("exe文件位于 dist 目录中")
        
        # 询问是否清理临时文件
        try:
            choice = input("\n是否清理构建临时文件？(y/n): ").lower().strip()
            if choice in ['y', 'yes', '是']:
                clean_build_files()
        except KeyboardInterrupt:
            print("\n操作取消")
    else:
        print("\n❌ 打包失败")


if __name__ == "__main__":
    main()
