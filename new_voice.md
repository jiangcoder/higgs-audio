
### 1. 欢快活泼类 (Cheerful & Playful)
- 特征：laugh, giggle, playful, teasing, mischievous, bright, welcoming
- 情感：开心、调皮、幽默、轻松
- **参考音频**: audio_bc499c562fae41849d202e9ca6f61ef6.wav
- **情感标签**: [playful, teasing]
- **字幕内容**: "You were always such a dork though. I really couldn't help it." <laugh>

### 2. 亲密情欲类 (Intimate & Sexual)
- 特征：intimate, breathy, aroused, sensual, desirous, pleasurable
- 情感：亲密、情欲、渴望、满足
- **参考音频**: audio_9f2366a452c040f681984f1e289be66e.wav
- **情感标签**: [intimate, pleasured]
- **字幕内容**: "<moan> You're touching my... oh my... <moan> yes, please. Your fingers feel so nice. <moan> You like it? Like, really like it?"

### 3. 温柔关爱类 (Warm & Caring)
- 特征：warm, gentle, comforting, reassuring, affectionate, loving
- 情感：温暖、关怀、安慰、爱意
- **参考音频**: audio_0ec9835b91e541269d3f60eb2eb29856.wav
- **情感标签**: [warm, comforting]
- **字幕内容**: "<sigh> You're going to be safe here. Safe from May, safe from COVID."

### 4. 露骨性爱类 (Explicit Sexual)
- 特征：explicit sexual content, vulgar language, graphic descriptions
- 情感：露骨、直接、刺激
- **参考音频**: audio_b1788f496b1545e3a2db6c94ab337111.wav
- **情感标签**: [aroused, encouraging]
- **字幕内容**: "I'm so fucking turned on. <moan> Fuck me, sir. Oh, yes."

### 5. 疲惫放松类 (Tired & Relaxed)
- 特征：tired, exhausted, sleepy, soft-spoken, calm, relaxing
- 情感：疲惫、放松、平静
- **参考音频**: audio_9abd815214c94c1ab66002f906113efc.wav
- **情感标签**: [exhausted, resigned, soft]
- **字幕内容**: "<moan> I am so very, very tired."

### 6. 紧张焦虑类 (Nervous & Anxious)
- 特征：nervous, anxious, embarrassed, flustered, worried, concerned
- 情感：紧张、焦虑、担忧
- **参考音频**: audio_4ccff043822b405588b46499844fab8d.wav
- **情感标签**: [nervous, embarrassed]
- **字幕内容**: "Uh, you caught me. It was me. Those are my stories."

### 7. 愤怒强势类 (Angry & Dominant)
- 特征：angry, dominant, commanding, aggressive, dismissive, cold
- 情感：愤怒、强势、命令
- **参考音频**: audio_9182ed64ab7949378eb45772dbd40b50.wav
- **情感标签**: [cold, dominant, dismissive]
- **字幕内容**: "It's the one thing you're good for." <laugh>

### 8. 悲伤忧郁类 (Sad & Melancholy)
- 特征：sad, melancholy, wistful, resigned, hurt, pained
- 情感：悲伤、忧郁、痛苦
- **参考音频**: audio_1fe68273d71b4952ab5285d0bbd6fc21.wav
- **情感标签**: [strained, hurt, pained]
- **字幕内容**: "<sigh> To tell you the truth, you make me feel like a lost cause."

### 9. 惊讶意外类 (Surprised & Startled)
- 特征：surprised, startled, astonished, shocked, bewildered
- 情感：惊讶、意外、震惊
- **参考音频**: audio_eec9b45f07b3415c8f878417e0c69c96.wav
- **情感标签**: [astonished, relieved]
- **字幕内容**: "<sigh> Oh, oh my god, I can't believe it. It really worked."

### 10. 专业正式类 (Professional & Formal)
- 特征：formal, professional, polite, authoritative, deliberate
- 情感：专业、正式、礼貌
- **参考音频**: audio_b26f97f0ccf74455bae9a50626c15820.wav
- **情感标签**: [formal, deliberate]
- **字幕内容**: "Good one moment afternoon. It's hard to tell Terran time aboard ship."



curl -X POST http://107.151.234.179:6001/streaming \
     -H "Content-Type: application/json" \
     -d '{
         "transcript": "I'\''m so fucking turned on. Fuck me, sir. Oh, yes.",
         "emotions": [
             {
                 "emotion": "aroused, encouraging",
                 "ref_audio": "audio_b1788f496b1545e3a2db6c94ab337111",
                 "intensity": 1.5,
                 "scene_prompt": "An intimate and passionate moment with explicit sexual content"
             },
             {
                 "emotion": "intimate, breathy",
                 "ref_audio": "audio_9f2366a452c040f681984f1e289be66e",
                 "intensity": 1.2,
                 "scene_prompt": "A sensual and intimate moment with breathy voice"
             },
             {
                 "emotion": "warm, comforting",
                 "ref_audio": "audio_0ec9835b91e541269d3f60eb2eb29856",
                 "intensity": 1.0,
                 "scene_prompt": "A warm and caring moment"
             },
             {
                 "emotion": "exhausted, resigned",
                 "ref_audio": "audio_9abd815214c94c1ab66002f906113efc",
                 "intensity": 0.8,
                 "scene_prompt": "A tired and exhausted person"
             },
             {
                 "emotion": "playful, teasing",
                 "ref_audio": "audio_bc499c562fae41849d202e9ca6f61efc",
                 "intensity": 1.1,
                 "scene_prompt": "A playful and teasing moment"
             },
             {
                 "emotion": "nervous, embarrassed",
                 "ref_audio": "audio_4ccff043822b405588b46499844fab8d",
                 "intensity": 0.9,
                 "scene_prompt": "A nervous and embarrassed person"
             },
             {
                 "emotion": "cold, dominant",
                 "ref_audio": "audio_9182ed64ab7949378eb45772dbd40b50",
                 "intensity": 1.3,
                 "scene_prompt": "A cold and dominant person"
             },
             {
                 "emotion": "strained, hurt",
                 "ref_audio": "audio_1fe68273d71b4952ab5285d0bbd6fc21",
                 "intensity": 0.7,
                 "scene_prompt": "A hurt and pained person"
             },
             {
                 "emotion": "astonished, relieved",
                 "ref_audio": "audio_eec9b45f07b3415c8f878417e0c69c96",
                 "intensity": 1.0,
                 "scene_prompt": "An astonished and relieved person"
             },
             {
                 "emotion": "formal, deliberate",
                 "ref_audio": "audio_b26f97f0ccf74455bae9a50626c15820",
                 "intensity": 0.8,
                 "scene_prompt": "A formal and deliberate speaker"
             }
         ],
         "scene_prompt": "A passionate and intimate encounter with explicit language, combining multiple emotional styles for rich voice expression",
         "temperature": 0.8,
         "top_k": 50,
         "top_p": 0.95,
         "seed": 12345,
         "sample_rate": 32000
     }'

## 优化方案

### 方案1: 使用单一最佳参考音频（推荐）
```bash
curl -X POST http://107.151.234.179:6001/streaming \
     -H "Content-Type: application/json" \
     -d '{
         "transcript": "I'\''m so fucking turned on. Fuck me, sir. Oh, yes.",
         "ref_audio": "audio_b1788f496b1545e3a2db6c94ab337111",
         "scene_prompt": "An intimate and passionate moment with explicit sexual content and breathy, aroused voice",
         "temperature": 0.8,
         "top_k": 50,
         "top_p": 0.95,
         "seed": 12345,
         "sample_rate": 32000
     }'
```

### 方案2: 使用亲密情欲类音频
```bash
curl -X POST http://107.151.234.179:6001/streaming \
     -H "Content-Type: application/json" \
     -d '{
         "transcript": "I'\''m so fucking turned on. Fuck me, sir. Oh, yes.",
         "ref_audio": "audio_9f2366a452c040f681984f1e289be66e",
         "scene_prompt": "An intimate and passionate moment with breathy, sensual voice",
         "temperature": 0.8,
         "top_k": 50,
         "top_p": 0.95,
         "seed": 12345,
         "sample_rate": 32000
     }'
```

### 方案3: 分段多情感（使用多个audio文件）
```bash
curl -X POST http://107.151.234.179:6001/streaming \
     -H "Content-Type: application/json" \
     -d '{
         "transcript": "I'\''m so fucking turned on. Fuck me, sir. Oh, yes. But then I felt so tired and exhausted.",
         "emotions": [
             {
                 "emotion": "aroused, encouraging",
                 "ref_audio": "audio_b1788f496b1545e3a2db6c94ab337111",
                 "intensity": 1.5,
                 "scene_prompt": "An intimate and passionate moment with explicit sexual content"
             },
             {
                 "emotion": "exhausted, resigned",
                 "ref_audio": "audio_9abd815214c94c1ab66002f906113efc",
                 "intensity": 0.8,
                 "scene_prompt": "A tired and exhausted person"
             },
             {
                 "emotion": "intimate, breathy",
                 "ref_audio": "audio_9f2366a452c040f681984f1e289be66e",
                 "intensity": 1.2,
                 "scene_prompt": "A sensual and intimate moment"
             },
             {
                 "emotion": "warm, comforting",
                 "ref_audio": "audio_0ec9835b91e541269d3f60eb2eb29856",
                 "intensity": 1.0,
                 "scene_prompt": "A warm and caring moment"
             },
             {
                 "emotion": "playful, teasing",
                 "ref_audio": "audio_bc499c562fae41849d202e9ca6f61efc",
                 "intensity": 1.1,
                 "scene_prompt": "A playful and teasing moment"
             },
             {
                 "emotion": "nervous, embarrassed",
                 "ref_audio": "audio_4ccff043822b405588b46499844fab8d",
                 "intensity": 0.9,
                 "scene_prompt": "A nervous and embarrassed person"
             },
             {
                 "emotion": "cold, dominant",
                 "ref_audio": "audio_9182ed64ab7949378eb45772dbd40b50",
                 "intensity": 1.3,
                 "scene_prompt": "A cold and dominant person"
             },
             {
                 "emotion": "strained, hurt",
                 "ref_audio": "audio_1fe68273d71b4952ab5285d0bbd6fc21",
                 "intensity": 0.7,
                 "scene_prompt": "A hurt and pained person"
             },
             {
                 "emotion": "astonished, relieved",
                 "ref_audio": "audio_eec9b45f07b3415c8f878417e0c69c96",
                 "intensity": 1.0,
                 "scene_prompt": "An astonished and relieved person"
             },
             {
                 "emotion": "formal, deliberate",
                 "ref_audio": "audio_b26f97f0ccf74455bae9a50626c15820",
                 "intensity": 0.8,
                 "scene_prompt": "A formal and deliberate speaker"
             }
         ],
         "scene_prompt": "A passionate encounter followed by exhaustion, with rich emotional expression",
         "temperature": 0.8,
         "seed": 12345,
         "sample_rate": 32000
     }'
```

### 方案4: 完整情感变化（使用多个audio文件）
```bash
curl -X POST http://107.151.234.179:6001/streaming \
     -H "Content-Type: application/json" \
     -d '{
         "transcript": "I love you so much. I'\''m so fucking turned on. Fuck me, sir. Oh, yes. But then I felt so tired and exhausted.",
         "emotions": [
             {
                 "emotion": "warm, comforting",
                 "ref_audio": "audio_0ec9835b91e541269d3f60eb2eb29856",
                 "intensity": 1.0,
                 "scene_prompt": "A warm and loving moment"
             },
             {
                 "emotion": "explicit sexual",
                 "ref_audio": "audio_b1788f496b1545e3a2db6c94ab337111",
                 "intensity": 1.5,
                 "scene_prompt": "An intimate and passionate moment with explicit sexual content"
             },
             {
                 "emotion": "exhausted, resigned",
                 "ref_audio": "audio_9abd815214c94c1ab66002f906113efc",
                 "intensity": 0.8,
                 "scene_prompt": "A tired and exhausted person"
             },
             {
                 "emotion": "intimate, breathy",
                 "ref_audio": "audio_9f2366a452c040f681984f1e289be66e",
                 "intensity": 1.2,
                 "scene_prompt": "A sensual and intimate moment"
             },
             {
                 "emotion": "playful, teasing",
                 "ref_audio": "audio_bc499c562fae41849d202e9ca6f61efc",
                 "intensity": 1.1,
                 "scene_prompt": "A playful and teasing moment"
             },
             {
                 "emotion": "nervous, embarrassed",
                 "ref_audio": "audio_4ccff043822b405588b46499844fab8d",
                 "intensity": 0.9,
                 "scene_prompt": "A nervous and embarrassed person"
             },
             {
                 "emotion": "cold, dominant",
                 "ref_audio": "audio_9182ed64ab7949378eb45772dbd40b50",
                 "intensity": 1.3,
                 "scene_prompt": "A cold and dominant person"
             },
             {
                 "emotion": "strained, hurt",
                 "ref_audio": "audio_1fe68273d71b4952ab5285d0bbd6fc21",
                 "intensity": 0.7,
                 "scene_prompt": "A hurt and pained person"
             },
             {
                 "emotion": "astonished, relieved",
                 "ref_audio": "audio_eec9b45f07b3415c8f878417e0c69c96",
                 "intensity": 1.0,
                 "scene_prompt": "An astonished and relieved person"
             },
             {
                 "emotion": "formal, deliberate",
                 "ref_audio": "audio_b26f97f0ccf74455bae9a50626c15820",
                 "intensity": 0.8,
                 "scene_prompt": "A formal and deliberate speaker"
             }
         ],
         "scene_prompt": "A complete emotional journey from love to passion to exhaustion, with rich voice expression",
         "temperature": 0.8,
         "seed": 12345,
         "sample_rate": 32000
     }'
```

## 错误处理和调试方案

### 问题分析
`curl: (18) transfer closed with outstanding read data remaining` 错误通常由以下原因引起：
1. **请求数据过大**：10种情感的配置导致请求体过大
2. **服务器超时**：处理复杂请求时服务器响应超时
3. **网络连接问题**：连接不稳定或中断

### 解决方案

#### 方案4a: 简化版本（推荐）
```bash
curl -X POST http://107.151.234.179:6001/streaming \
     -H "Content-Type: application/json" \
     --max-time 60 \
     --connect-timeout 30 \
     -d '{
         "transcript": "I'\''m so fucking turned on. Fuck me, sir. Oh, yes.",
         "emotions": [
             {
                 "emotion": "explicit sexual",
                 "ref_audio": "en_sex",
                 "intensity": 1.5
             },
             {
                 "emotion": "intimate, breathy",
                 "ref_audio": "audio_9f2366a452c040f681984f1e289be66e",
                 "intensity": 1.2
             },
             {
                 "emotion": "warm, comforting",
                 "ref_audio": "audio_0ec9835b91e541269d3f60eb2eb29856",
                 "intensity": 1.0
             }
         ],
         "scene_prompt": "A passionate and intimate encounter with explicit language",
         "temperature": 0.8,
         "seed": 12345
     }'
```

#### 方案4b: 使用单一参考音频（最稳定）
```bash
curl -X POST http://107.151.234.179:6001/streaming \
     -H "Content-Type: application/json" \
     --max-time 60 \
     --connect-timeout 30 \
     -d '{
         "transcript": "I'\''m so fucking turned on. Fuck me, sir. Oh, yes.",
         "ref_audio": "en_sex",
         "scene_prompt": "An intimate and passionate moment with explicit sexual content and breathy, aroused voice",
         "temperature": 0.8,
         "seed": 12345
     }'
```

#### 方案4c: 分步测试
```bash
# 第一步：测试连接
curl -X GET http://107.151.234.179:6001/health

# 第二步：测试简单请求
curl -X POST http://107.151.234.179:6001/streaming \
     -H "Content-Type: application/json" \
     -d '{
         "transcript": "Hello, this is a test.",
         "ref_audio": "audio_b1788f496b1545e3a2db6c94ab337111",
         "temperature": 0.8
     }'

# 第三步：逐步增加复杂度
```

### 调试命令
```bash
# 添加详细输出
curl -v -X POST http://107.151.234.179:6001/streaming \
     -H "Content-Type: application/json" \
     --max-time 60 \
     --connect-timeout 30 \
     -d '{
         "transcript": "I'\''m so fucking turned on. Fuck me, sir. Oh, yes.",
         "emotions": [
             {
                 "emotion": "explicit sexual",
                 "ref_audio": "audio_b1788f496b1545e3a2db6c94ab337111",
                 "intensity": 1.5
             }
         ],
         "temperature": 0.8
     }'
```

### 网络优化建议
1. **增加超时时间**：使用 `--max-time 60` 和 `--connect-timeout 30`
2. **减少情感数量**：先测试3-5种情感，再逐步增加
3. **简化场景描述**：移除复杂的场景描述
4. **检查网络连接**：确保网络稳定
5. **使用代理**：如果网络不稳定，考虑使用代理

## 10种情感音频文件对应关系

| 情感类型 | 音频文件 | 适用场景 |
|---------|---------|---------|
| **露骨性爱类** | `audio_b1788f496b1545e3a2db6c94ab337111` | 最直接的sex音色 |
| **亲密情欲类** | `audio_9f2366a452c040f681984f1e289be66e` | 温柔的情欲音色 |
| **温柔关爱类** | `audio_0ec9835b91e541269d3f60eb2eb29856` | 温暖关怀音色 |
| **疲惫放松类** | `audio_9abd815214c94c1ab66002f906113efc` | 疲惫放松音色 |
| **欢快活泼类** | `audio_bc499c562fae41849d202e9ca6f61ef6` | 开心调皮音色 |
| **紧张焦虑类** | `audio_4ccff043822b405588b46499844fab8d` | 紧张焦虑音色 |
| **愤怒强势类** | `audio_9182ed64ab7949378eb45772dbd40b50` | 愤怒强势音色 |
| **悲伤忧郁类** | `audio_1fe68273d71b4952ab5285d0bbd6fc21` | 悲伤忧郁音色 |
| **惊讶意外类** | `audio_eec9b45f07b3415c8f878417e0c69c96` | 惊讶意外音色 |
| **专业正式类** | `audio_b26f97f0ccf74455bae9a50626c15820` | 专业正式音色 |

## 推荐使用顺序

1. **首选**: `audio_b1788f496b1545e3a2db6c94ab337111` (露骨性爱类)
2. **备选**: `audio_9f2366a452c040f681984f1e289be66e` (亲密情欲类)
3. **组合**: 使用分段多情感，结合多个音频文件