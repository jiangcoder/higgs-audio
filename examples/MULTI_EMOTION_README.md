# 多情感流式音频生成接口

这个文档介绍了如何使用支持多情感的流式音频生成接口。

## 功能特性

- **多情感支持**: 支持在单次请求中配置多种情感状态
- **参考音频映射**: 每种情感可以对应不同的参考音频
- **情感强度控制**: 可以调节每种情感的强度
- **场景描述**: 为每种情感提供详细的场景描述
- **流式生成**: 支持实时流式音频生成
- **多语言支持**: 支持中英文等多种语言

## 接口说明

### 请求格式

```json
{
  "transcript": "要转换的文本内容",
  "emotions": [
    {
      "emotion": "情感名称",
      "ref_audio": "参考音频名称",
      "intensity": 1.0,
      "scene_prompt": "情感场景描述"
    }
  ],
  "scene_prompt": "整体场景描述",
  "temperature": 1.0,
  "top_k": 50,
  "top_p": 0.95,
  "seed": 12345,
  "sample_rate": 32000
}
```

### 参数说明

#### 基础参数
- `transcript`: 要转换为语音的文本内容
- `scene_prompt`: 整体场景描述（可选）
- `temperature`: 采样温度，控制生成的随机性
- `top_k`: Top-k过滤参数
- `top_p`: Top-p过滤参数
- `seed`: 随机种子，用于结果复现
- `sample_rate`: 输出音频的采样率

#### 多情感参数
- `emotions`: 情感配置列表
  - `emotion`: 情感名称（如 "happy", "sad", "angry", "excited" 等）
  - `ref_audio`: 该情感对应的参考音频名称（需要在 voice_prompts 目录中存在）
  - `intensity`: 情感强度，范围 0.0-2.0
  - `scene_prompt`: 该情感的场景描述（可选）

## 使用方法

### 1. 启动服务器

```bash
cd examples
python serve_fastapi.py
```

服务器将在 `http://localhost:6001` 启动。

### 2. 基础多情感生成

```python
import requests
import json

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

### 3. 使用示例脚本

运行提供的示例脚本：

```bash
python multi_emotion_example.py
```

这个脚本包含了三个示例：
- 基础多情感生成
- 情感分段生成  
- 中文多情感生成

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

### 角色音频
- `shrek_shrek`: 怪物声音
- `shrek_fiona`: 公主声音
- `shrek_donkey`: 驴子声音

## 情感配置建议

### 情感类型
- `happy/joyful`: 快乐、愉悦
- `sad/melancholy`: 悲伤、忧郁
- `angry/furious`: 愤怒、激动
- `calm/peaceful`: 平静、冷静
- `excited/enthusiastic`: 兴奋、热情
- `serious/formal`: 严肃、正式
- `friendly/warm`: 友好、温暖

### 强度设置
- `0.0-0.5`: 轻微情感
- `0.5-1.0`: 中等情感
- `1.0-1.5`: 强烈情感
- `1.5-2.0`: 非常强烈的情感

### 场景描述示例
```
"A cheerful person sharing good news with friends"
"A person expressing deep disappointment about a failed project"
"An enthusiastic speaker celebrating a major achievement"
"A calm and composed professional explaining complex concepts"
```

## 高级用法

### 1. 情感分段

可以通过文本分段来实现更精确的情感控制：

```python
emotion_segments = [
    {
        "start": 0,
        "end": 20,
        "emotion_index": 0,  # 使用第一个情感配置
        "emotion_config": {
            "emotion": "happy",
            "ref_audio": "belinda",
            "intensity": 1.2
        }
    },
    {
        "start": 20,
        "end": 40,
        "emotion_index": 1,  # 使用第二个情感配置
        "emotion_config": {
            "emotion": "sad",
            "ref_audio": "broom_salesman", 
            "intensity": 0.8
        }
    }
]
```

### 2. 混合策略

结合参考音频和场景描述：

```python
# 系统消息包含情感场景
request_data = {
    "transcript": "文本内容",
    "emotions": emotions_config,
    "scene_prompt": "详细的场景描述",
    "ref_audio_in_system_message": True
}
```

## 错误处理

### 常见错误

1. **参考音频不存在**
   ```
   Error: Voice prompt audio file /path/to/voice_prompts/audio_name.wav does not exist.
   ```
   解决：确保参考音频文件存在于 `voice_prompts` 目录中

2. **模型未加载**
   ```
   Error: Model is not loaded yet.
   ```
   解决：等待服务器启动完成，模型加载需要一些时间

3. **内存不足**
   ```
   Error: CUDA out of memory
   ```
   解决：减少 `max_new_tokens` 或使用更小的模型

### 调试技巧

1. 启用详细日志：
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. 检查音频文件：
   ```python
   import os
   print(os.path.exists("examples/voice_prompts/belinda.wav"))
   ```

3. 验证请求格式：
   ```python
   import json
   print(json.dumps(request_data, indent=2))
   ```

## 性能优化

### 1. 缓存策略
- 使用相同的 `seed` 值可以获得可重现的结果
- 对于相同的文本和情感配置，可以缓存结果

### 2. 批量处理
- 对于多个请求，可以考虑批量处理
- 使用 `generation_chunk_buffer_size` 控制内存使用

### 3. 参数调优
- 降低 `temperature` 可以获得更稳定的结果
- 调整 `top_k` 和 `top_p` 可以控制生成质量

## 扩展开发

### 添加新的参考音频

1. 在 `voice_prompts` 目录中添加音频文件：
   ```
   voice_prompts/
   ├── new_voice.wav
   └── new_voice.txt
   ```

2. 在 `profile.yaml` 中添加描述：
   ```yaml
   profiles:
     new_voice: "描述这个声音的特点"
   ```

### 自定义情感类型

可以在代码中扩展支持的情感类型：

```python
EMOTION_TYPES = {
    "happy": "快乐",
    "sad": "悲伤", 
    "angry": "愤怒",
    "calm": "平静",
    "excited": "兴奋",
    "serious": "严肃",
    "friendly": "友好"
}
```

## 注意事项

1. **音频质量**: 参考音频的质量直接影响生成结果的质量
2. **情感一致性**: 确保情感配置与文本内容匹配
3. **内存使用**: 多情感生成会消耗更多内存
4. **生成时间**: 复杂的情感配置会增加生成时间
5. **语言支持**: 不同语言的文本可能需要不同的情感配置

## 技术支持

如果遇到问题，请检查：
1. 服务器是否正常启动
2. 参考音频文件是否存在
3. 请求格式是否正确
4. 网络连接是否正常

更多信息请参考项目文档和示例代码。 