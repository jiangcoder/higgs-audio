#!/usr/bin/env python3
"""
多情感流式音频生成服务器启动脚本

这个脚本用于快速启动支持多情感的流式音频生成服务器。
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path


def check_dependencies():
    """检查依赖项"""
    print("检查依赖项...")
    
    required_packages = [
        "torch",
        "transformers", 
        "fastapi",
        "uvicorn",
        "librosa",
        "soundfile",
        "numpy",
        "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (缺失)")
    
    if missing_packages:
        print(f"\n缺少以下依赖项: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✓ 所有依赖项已安装")
    return True


def check_voice_prompts():
    """检查语音提示文件"""
    print("\n检查语音提示文件...")
    
    voice_prompts_dir = Path("voice_prompts")
    if not voice_prompts_dir.exists():
        print("✗ voice_prompts 目录不存在")
        return False
    
    # 检查一些常用的语音文件
    required_files = [
        "belinda.wav",
        "belinda.txt", 
        "broom_salesman.wav",
        "broom_salesman.txt",
        "en_man.wav",
        "en_man.txt",
        "en_woman.wav", 
        "en_woman.txt"
    ]
    
    missing_files = []
    
    for file in required_files:
        file_path = voice_prompts_dir / file
        if file_path.exists():
            print(f"✓ {file}")
        else:
            missing_files.append(file)
            print(f"✗ {file} (缺失)")
    
    if missing_files:
        print(f"\n缺少以下语音文件: {', '.join(missing_files)}")
        print("请确保 voice_prompts 目录包含所需的音频和文本文件")
        return False
    
    print("✓ 语音提示文件检查通过")
    return True


def start_server():
    """启动服务器"""
    print("\n启动多情感流式音频生成服务器...")
    
    # 检查当前目录
    current_dir = Path.cwd()
    if current_dir.name != "examples":
        print("请确保在 examples 目录中运行此脚本")
        return False
    
    # 启动服务器
    try:
        print("正在启动服务器...")
        print("服务器将在 http://localhost:6001 启动")
        print("按 Ctrl+C 停止服务器")
        print("-" * 50)
        
        # 使用 subprocess 启动服务器
        process = subprocess.Popen([
            sys.executable, "serve_fastapi.py"
        ])
        
        # 等待服务器启动
        print("等待服务器启动...")
        time.sleep(10)
        
        # 检查服务器是否正常启动
        try:
            response = requests.get("http://localhost:6001/docs", timeout=5)
            if response.status_code == 200:
                print("✓ 服务器启动成功!")
                print("📖 API文档: http://localhost:6001/docs")
                print("🔧 测试接口: http://localhost:6001/streaming")
                print("\n现在可以运行测试脚本:")
                print("python test_multi_emotion.py")
                print("\n或者运行示例脚本:")
                print("python multi_emotion_example.py")
            else:
                print("✗ 服务器启动失败")
                return False
        except requests.exceptions.RequestException:
            print("✗ 无法连接到服务器")
            return False
        
        # 等待用户中断
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n正在停止服务器...")
            process.terminate()
            process.wait()
            print("服务器已停止")
        
        return True
        
    except Exception as e:
        print(f"启动服务器时出错: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("多情感流式音频生成服务器")
    print("=" * 60)
    
    # 检查依赖项
    if not check_dependencies():
        return False
    
    # 检查语音提示文件
    if not check_voice_prompts():
        return False
    
    # 启动服务器
    return start_server()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 