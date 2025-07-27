#!/usr/bin/env python3
"""
多情感音频生成示例脚本

这个脚本演示了如何使用支持多情感的流式音频生成接口。
"""

import requests
import json
import base64
import io
import wave
import numpy as np
from typing import List, Dict, Any


def create_multi_emotion_request(
    transcript: str,
    emotions: List[Dict[str, Any]],
    scene_prompt: str = None,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: int = 12345
) -> Dict[str, Any]:
    """
    创建多情感音频生成请求
    
    Args:
        transcript: 要转换的文本
        emotions: 情感配置列表
        scene_prompt: 场景描述
        temperature: 采样温度
        top_k: Top-k过滤
        top_p: Top-p过滤
        seed: 随机种子
    
    Returns:
        请求字典
    """
    return {
        "transcript": transcript,
        "emotions": emotions,
        "scene_prompt": scene_prompt,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed,
        "sample_rate": 32000
    }


def decode_audio_from_base64(audio_b64: str) -> np.ndarray:
    """
    从base64编码的音频数据解码为numpy数组
    
    Args:
        audio_b64: base64编码的音频数据
    
    Returns:
        音频numpy数组
    """
    audio_bytes = base64.b64decode(audio_b64)
    audio_buffer = io.BytesIO(audio_bytes)
    
    with wave.open(audio_buffer, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32767.0
    
    return audio_array


def save_audio_to_wav(audio_array: np.ndarray, filename: str, sample_rate: int = 32000):
    """
    保存音频数组为WAV文件
    
    Args:
        audio_array: 音频numpy数组
        filename: 输出文件名
        sample_rate: 采样率
    """
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"音频已保存到: {filename}")


def stream_audio_generation(server_url: str, request_data: Dict[str, Any]) -> np.ndarray:
    """
    流式音频生成
    
    Args:
        server_url: 服务器URL
        request_data: 请求数据
    
    Returns:
        生成的音频数组
    """
    print("开始流式音频生成...")
    
    response = requests.post(
        f"{server_url}/streaming",
        json=request_data,
        stream=True,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code != 200:
        raise Exception(f"请求失败: {response.status_code} - {response.text}")
    
    audio_b64 = None
    
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
                            print("收到音频数据")
                            break
                except json.JSONDecodeError:
                    continue
    
    if audio_b64 is None:
        raise Exception("未收到音频数据")
    
    # 解码音频
    audio_array = decode_audio_from_base64(audio_b64)
    print(f"音频生成完成，长度: {len(audio_array)} 采样点")
    
    return audio_array


def main():
    """主函数 - 演示多情感音频生成"""
    
    # 服务器配置
    SERVER_URL = "http://localhost:6001"
    
    # 示例1: 基础多情感生成
    print("=== 示例1: 基础多情感生成 ===")
    
    emotions_config = [
        {
            "emotion": "happy",
            "ref_audio": "belinda",
            "intensity": 1.2,
            "scene_prompt": "A cheerful person sharing good news"
        },
        {
            "emotion": "sad",
            "ref_audio": "broom_salesman",
            "intensity": 0.8,
            "scene_prompt": "A person expressing disappointment"
        },
        {
            "emotion": "excited",
            "ref_audio": "en_woman",
            "intensity": 1.5,
            "scene_prompt": "An enthusiastic person celebrating success"
        }
    ]
    
    transcript = "I'm so happy to see you! But then I heard the bad news and felt really sad. However, when I found out about the surprise, I became incredibly excited!"
    
    request_data = create_multi_emotion_request(
        transcript=transcript,
        emotions=emotions_config,
        scene_prompt="A conversation with mixed emotions",
        temperature=0.8,
        seed=42
    )
    
    try:
        audio_array = stream_audio_generation(SERVER_URL, request_data)
        save_audio_to_wav(audio_array, "multi_emotion_example1.wav")
    except Exception as e:
        print(f"示例1失败: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例2: 情感分段生成
    print("=== 示例2: 情感分段生成 ===")
    
    emotions_config_2 = [
        {
            "emotion": "calm",
            "ref_audio": "en_man",
            "intensity": 1.0,
            "scene_prompt": "A calm and composed speaker"
        },
        {
            "emotion": "angry",
            "ref_audio": "mabaoguo",
            "intensity": 1.3,
            "scene_prompt": "An angry person expressing frustration"
        }
    ]
    
    transcript_2 = "Let me explain this calmly. The situation is quite simple. But I'm really frustrated with how this has been handled!"
    
    request_data_2 = create_multi_emotion_request(
        transcript=transcript_2,
        emotions=emotions_config_2,
        scene_prompt="A person transitioning from calm to angry",
        temperature=0.7,
        seed=123
    )
    
    try:
        audio_array_2 = stream_audio_generation(SERVER_URL, request_data_2)
        save_audio_to_wav(audio_array_2, "multi_emotion_example2.wav")
    except Exception as e:
        print(f"示例2失败: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例3: 中文多情感生成
    print("=== 示例3: 中文多情感生成 ===")
    
    emotions_config_3 = [
        {
            "emotion": "joyful",
            "ref_audio": "zh_man_sichuan",
            "intensity": 1.1,
            "scene_prompt": "一个快乐的人分享好消息"
        },
        {
            "emotion": "serious",
            "ref_audio": "en_man",
            "intensity": 0.9,
            "scene_prompt": "一个严肃的人讨论重要问题"
        }
    ]
    
    transcript_3 = "今天真是太开心了！我收到了一个好消息。不过接下来要讨论的事情很重要，需要认真对待。"
    
    request_data_3 = create_multi_emotion_request(
        transcript=transcript_3,
        emotions=emotions_config_3,
        scene_prompt="一个人从开心到严肃的情绪转换",
        temperature=0.9,
        seed=456
    )
    
    try:
        audio_array_3 = stream_audio_generation(SERVER_URL, request_data_3)
        save_audio_to_wav(audio_array_3, "multi_emotion_example3.wav")
    except Exception as e:
        print(f"示例3失败: {e}")


if __name__ == "__main__":
    main() 