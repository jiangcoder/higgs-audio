#!/usr/bin/env python3
"""
多情感音频生成功能测试脚本

这个脚本用于测试多情感音频生成功能是否正常工作。
"""

import requests
import json
import time
from typing import Dict, Any


def test_server_health(server_url: str) -> bool:
    """测试服务器健康状态"""
    try:
        response = requests.get(f"{server_url}/docs", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"服务器连接失败: {e}")
        return False


def test_basic_generation(server_url: str) -> bool:
    """测试基础音频生成"""
    print("测试基础音频生成...")
    
    request_data = {
        "transcript": "Hello, this is a test.",
        "ref_audio": "en_man",
        "temperature": 0.8,
        "seed": 123
    }
    
    try:
        response = requests.post(
            f"{server_url}/generate",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✓ 基础音频生成测试通过")
            return True
        else:
            print(f"✗ 基础音频生成测试失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ 基础音频生成测试异常: {e}")
        return False


def test_multi_emotion_generation(server_url: str) -> bool:
    """测试多情感音频生成"""
    print("测试多情感音频生成...")
    
    emotions_config = [
        {
            "emotion": "happy",
            "ref_audio": "belinda",
            "intensity": 1.0,
            "scene_prompt": "A happy person"
        },
        {
            "emotion": "sad",
            "ref_audio": "broom_salesman", 
            "intensity": 0.8,
            "scene_prompt": "A sad person"
        }
    ]
    
    request_data = {
        "transcript": "I'm happy to see you! But I'm also feeling sad.",
        "emotions": emotions_config,
        "scene_prompt": "Mixed emotions",
        "temperature": 0.8,
        "seed": 123
    }
    
    try:
        response = requests.post(
            f"{server_url}/generate",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✓ 多情感音频生成测试通过")
            return True
        else:
            print(f"✗ 多情感音频生成测试失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ 多情感音频生成测试异常: {e}")
        return False


def test_streaming_generation(server_url: str) -> bool:
    """测试流式音频生成"""
    print("测试流式音频生成...")
    
    emotions_config = [
        {
            "emotion": "excited",
            "ref_audio": "en_woman",
            "intensity": 1.2,
            "scene_prompt": "An excited person"
        }
    ]
    
    request_data = {
        "transcript": "This is amazing! I can't believe it!",
        "emotions": emotions_config,
        "temperature": 0.8,
        "seed": 456
    }
    
    try:
        response = requests.post(
            f"{server_url}/streaming",
            json=request_data,
            stream=True,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"✗ 流式音频生成测试失败: {response.status_code}")
            return False
        
        audio_received = False
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        data_str = line_str[6:]  # 移除 'data: ' 前缀
                        data = json.loads(data_str)
                        
                        if 'choices' in data and len(data['choices']) > 0:
                            choice = data['choices'][0]
                            if 'delta' in choice and 'audio' in choice['delta']:
                                audio_b64 = choice['delta']['audio']['data']
                                if audio_b64:
                                    audio_received = True
                                    break
                    except json.JSONDecodeError:
                        continue
        
        if audio_received:
            print("✓ 流式音频生成测试通过")
            return True
        else:
            print("✗ 流式音频生成测试失败: 未收到音频数据")
            return False
            
    except Exception as e:
        print(f"✗ 流式音频生成测试异常: {e}")
        return False


def test_emotion_segments(server_url: str) -> bool:
    """测试情感分段功能"""
    print("测试情感分段功能...")
    
    emotions_config = [
        {
            "emotion": "happy",
            "ref_audio": "belinda",
            "intensity": 1.2,
            "scene_prompt": "A happy person"
        },
        {
            "emotion": "sad",
            "ref_audio": "broom_salesman",
            "intensity": 0.8,
            "scene_prompt": "A sad person"
        }
    ]
    
    emotion_segments = [
        {"start": 0, "end": 20, "emotion_index": 0},  # "I'm happy to see you!" -> happy
        {"start": 20, "end": 50, "emotion_index": 1}   # "But I'm also feeling sad." -> sad
    ]
    
    request_data = {
        "transcript": "I'm happy to see you! But I'm also feeling sad.",
        "emotions": emotions_config,
        "emotion_segments": emotion_segments,
        "scene_prompt": "Emotion transition",
        "temperature": 0.8,
        "seed": 789
    }
    
    try:
        response = requests.post(
            f"{server_url}/generate",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✓ 情感分段功能测试通过")
            return True
        else:
            print(f"✗ 情感分段功能测试失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ 情感分段功能测试异常: {e}")
        return False


def main():
    """主测试函数"""
    SERVER_URL = "http://localhost:6001"
    
    print("开始多情感音频生成功能测试...")
    print("=" * 50)
    
    # 测试服务器健康状态
    if not test_server_health(SERVER_URL):
        print("服务器未启动或无法连接，请先启动服务器")
        return
    
    print("服务器连接正常")
    print("-" * 30)
    
    # 运行所有测试
    tests = [
        ("基础音频生成", test_basic_generation),
        ("多情感音频生成", test_multi_emotion_generation),
        ("流式音频生成", test_streaming_generation),
        ("情感分段功能", test_emotion_segments),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func(SERVER_URL):
            passed += 1
        time.sleep(1)  # 避免请求过于频繁
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！多情感功能正常工作。")
    else:
        print("❌ 部分测试失败，请检查相关功能。")


if __name__ == "__main__":
    main() 