# 多情感音频生成 - 简化版

## 概述

这是一个简化版的多情感音频生成实现，去掉了复杂的情感分段功能，保持简单易用。

## 功能特性

- ✅ **多情感支持**: 支持在单次请求中配置多种情感状态
- ✅ **参考音频映射**: 每种情感可以对应不同的参考音频
- ✅ **情感强度控制**: 可以调节每种情感的强度（0.0-2.0）
- ✅ **场景描述**: 为每种情感提供详细的场景描述
- ✅ **流式生成**: 支持实时流式音频生成
- ✅ **多语言支持**: 支持中英文等多种语言

## 使用方法

### 1. 基础多情感生成

```python
import requests

# 配置多种情感
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
    }
]

# 创建请求
request_data = {
    "transcript": "I'm so happy to see you! But then I heard the bad news and felt really sad.",
    "emotions": emotions_config,
    "scene_prompt": "A conversation with mixed emotions",
    "temperature": 0.8,
    "seed": 42
}

# 发送请求
response = requests.post(
    "http://localhost:6001/streaming",
    json=request_data,
    stream=True
)
```

### 2. 流式生成

```python
# 处理流式响应
for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8')[6:])  # 移除 'data: ' 前缀
        if 'choices' in data and len(data['choices']) > 0:
            choice = data['choices'][0]
            if 'delta' in choice and 'audio' in choice['delta']:
                audio_b64 = choice['delta']['audio']['data']
                # 处理音频数据...
                break
```

## 参数说明

### 基础参数
- `transcript`: 要转换为语音的文本内容
- `scene_prompt`: 整体场景描述（可选）
- `temperature`: 采样温度，控制生成的随机性
- `top_k`: Top-k过滤参数
- `top_p`: Top-p过滤参数
- `seed`: 随机种子，用于结果复现
- `sample_rate`: 输出音频的采样率

### 多情感参数
- `emotions`: 情感配置列表
  - `emotion`: 情感名称（如 "happy", "sad", "angry", "excited" 等）
  - `ref_audio`: 该情感对应的参考音频名称
  - `intensity`: 情感强度，范围 0.0-2.0
  - `scene_prompt`: 该情感的场景描述（可选）

## 工作原理

1. **多情感上下文准备**: 系统会为每种情感准备参考音频示例
2. **情感标签处理**: 文本会被添加情感标签，帮助模型理解情感上下文
3. **模型生成**: 模型会根据提供的多种情感参考，生成包含相应情感特征的音频
4. **流式输出**: 支持实时流式音频生成

## 示例脚本

运行提供的示例脚本：

```bash
python examples/multi_emotion_example.py
```

这个脚本包含了三个示例：
- 基础多情感生成
- 情感分段生成  
- 中文多情感生成

## 测试验证

运行测试脚本验证功能：

```bash
python examples/test_multi_emotion.py
```

## 可用的参考音频

在 `voice_prompts` 目录中提供了多种参考音频：

### 英文音频
- `belinda`: 女性声音，适合快乐、兴奋的情感
- `broom_salesman`: 男性声音，适合严肃、悲伤的情感
- `en_man`: 男性声音，适合冷静、专业的情感
- `en_woman`: 女性声音，适合温柔、亲切的情感

### 中文音频
- `zh_man_sichuan`: 四川口音男性声音
- `mabaoguo`: 男性声音，适合愤怒、激动的情感

## 注意事项

1. **简化设计**: 当前实现使用第一个情感作为主要情感，其他情感作为参考
2. **模型理解**: 模型会根据提供的多种情感参考，自动选择合适的风格
3. **情感过渡**: 情感之间的过渡是自然的，不需要手动分段
4. **性能**: 简化后的实现性能更好，更容易使用

## 总结

这个简化版的多情感实现保持了核心功能，去掉了复杂的情感分段，使得使用更加简单直观。对于大多数应用场景，这个实现已经足够使用。 