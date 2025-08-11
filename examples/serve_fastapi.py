# -*- coding: utf-8 -*-
import io
import time
import re
import os
import copy
import sys
from typing import List, Literal, Optional, Dict, Any

# 添加上级目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import soundfile as sf
import langid
import jieba
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger
import librosa
import threading
import base64
import json
import asyncio
import uuid  # 添加UUID支持用于请求标识

from boson_multimodal.data_types import Message, AudioContent, TextContent, ChatMLSample
from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern, streaming_revert_delay_pattern
from boson_multimodal.serve.serve_engine import AsyncHiggsAudioStreamer
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache
from dataclasses import asdict

# Re-implementing necessary components from generation.py

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"
MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


class EmotionConfig(BaseModel):
    """情感配置模型"""
    emotion: str = Field(..., description="情感名称，如 'happy', 'sad', 'angry', 'excited'")
    ref_audio: str = Field(..., description="该情感对应的参考音频名称")
    intensity: float = Field(1.0, ge=0.0, le=2.0, description="情感强度，0.0-2.0")
    scene_prompt: Optional[str] = Field(None, description="该情感的场景描述")


class HiggsAudioModelClient:
    def __init__(
        self,
        model_path,
        audio_tokenizer,
        device_id=None,
        kv_cache_lengths: List[int] = [1024, 2048],
        use_static_kv_cache=False,
    ):
        if device_id is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = f"cuda:{device_id}"
        self._audio_tokenizer = (
            load_higgs_audio_tokenizer(audio_tokenizer, device=self._device)
            if isinstance(audio_tokenizer, str)
            else audio_tokenizer
        )
        self._model = HiggsAudioModel.from_pretrained(
            model_path,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()
        self._kv_cache_lengths = kv_cache_lengths
        self._use_static_kv_cache = use_static_kv_cache

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
        self._collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self._config.audio_in_token_idx,
            audio_out_token_id=self._config.audio_out_token_idx,
            audio_stream_bos_id=self._config.audio_stream_bos_id,
            audio_stream_eos_id=self._config.audio_stream_eos_id,
            encode_whisper_embed=self._config.encode_whisper_embed,
            pad_token_id=self._config.pad_token_id,
            return_audio_in_tokens=self._config.encode_audio_in_tokens,
            use_delay_pattern=self._config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self._config.audio_num_codebooks,
        )
        
        # 移除实例级别的KV cache，改为每个请求创建独立的cache
        self._use_static_kv_cache = use_static_kv_cache
        
        # 为并发访问添加锁，但只保护模型结构的修改，不阻塞推理
        self._model_lock = threading.RLock()
        
        # 如果启用静态缓存，预先配置缓存参数
        if use_static_kv_cache:
            self._cache_config = self._prepare_cache_config()

    def _prepare_cache_config(self):
        """准备缓存配置，但不创建实际缓存实例"""
        cache_config = copy.deepcopy(self._model.config.text_config)
        cache_config.num_hidden_layers = self._model.config.text_config.num_hidden_layers
        if self._model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self._model.config.audio_dual_ffn_layers)
        return cache_config

    def _create_kv_caches_for_request(self):
        """为单个请求创建独立的KV缓存"""
        if not self._use_static_kv_cache:
            return None
            
        kv_caches = {
            length: StaticCache(
                config=self._cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self._model.device,
                dtype=self._model.dtype,
            )
            for length in sorted(self._kv_cache_lengths)
        }
        return kv_caches

    def _prepare_kv_caches(self, kv_caches):
        """重置指定的KV缓存"""
        if kv_caches:
            for kv_cache in kv_caches.values():
                kv_cache.reset()

    @torch.inference_mode()
    def generate(
        self,
        messages,
        audio_ids,
        chunked_text,
        generation_chunk_buffer_size,
        max_new_tokens=2048,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        seed=123,
        request_id=None,  # 添加请求ID参数
        **kwargs,
    ):
        # 为当前请求创建独立的KV缓存
        request_kv_caches = self._create_kv_caches_for_request()
        request_id = request_id or str(uuid.uuid4())[:8]
        
        logger.info(f"[Request {request_id}] Starting generation with independent KV cache")
        
        sr = 32000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []
        for idx, chunk_text in enumerate(chunked_text):
            generation_messages.append(Message(role="user", content=chunk_text))

            chatml_sample_for_tokenization = ChatMLSample(messages=messages + generation_messages)
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample_for_tokenization, self._tokenizer)

            postfix = self._tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
            input_tokens.extend(postfix)

            context_audio_ids = audio_ids + generated_audio_ids

            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1) if context_audio_ids else None,
                audio_ids_start=torch.cumsum(torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0) if context_audio_ids else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            batch_data = self._collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)
            
            # 使用请求独立的KV缓存
            if self._use_static_kv_cache:
                self._prepare_kv_caches(request_kv_caches)

            outputs = self._model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=request_kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self._tokenizer,
                seed=seed,
            )
            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                logger.debug(f"[Request {request_id}] audio output shape: {audio_out_ids.shape}")
                if self._config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                audio_out_ids = audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1]
                step_audio_out_ids_l.append(audio_out_ids)

            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)

            generation_messages.append(Message(role="assistant", content=AudioContent(audio_url="")))
            if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
                generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
                generation_messages = generation_messages[(-2 * generation_chunk_buffer_size):]

        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)
        logger.info(f"[Request {request_id}] Final output shape: {concat_audio_out_ids.shape}")
        concat_wv = self._audio_tokenizer.decode(concat_audio_out_ids.unsqueeze(0))[0, 0]
        return concat_wv, sr

    @torch.inference_mode()
    def stream_generate(
        self,
        messages,
        audio_ids,
        chunked_text,
        generation_chunk_buffer_size,
        max_new_tokens=2048,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        seed=123,
        streamer=None,
        request_id=None,  # 添加请求ID参数
        **kwargs,
    ):
        # 为当前请求创建独立的KV缓存
        request_kv_caches = self._create_kv_caches_for_request()
        request_id = request_id or str(uuid.uuid4())[:8]
        
        logger.info(f"[Request {request_id}] Starting streaming generation with independent KV cache")
        
        sr = 32000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []

        for idx, chunk_text in enumerate(chunked_text):
            generation_messages.append(Message(role="user", content=chunk_text))

            chatml_sample_for_tokenization = ChatMLSample(messages=messages + generation_messages)
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample_for_tokenization, self._tokenizer)

            postfix = self._tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
            input_tokens.extend(postfix)

            context_audio_ids = audio_ids + generated_audio_ids

            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1) if context_audio_ids else None,
                audio_ids_start=torch.cumsum(torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0) if context_audio_ids else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            batch_data = self._collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)
            
            # 使用请求独立的KV缓存
            if self._use_static_kv_cache:
                self._prepare_kv_caches(request_kv_caches)

            outputs = self._model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=request_kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self._tokenizer,
                seed=seed,
                streamer=streamer,
            )

            if streamer is None:
                step_audio_out_ids_l = []
                for ele in outputs[1]:
                    audio_out_ids = ele
                    if self._config.use_delay_pattern:
                        audio_out_ids = revert_delay_pattern(audio_out_ids)
                    step_audio_out_ids_l.append(audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1])
                audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
                audio_out_ids_l.append(audio_out_ids)
                generated_audio_ids.append(audio_out_ids)

                generation_messages.append(Message(role="assistant", content=AudioContent(audio_url="")))
                if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
                    generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
                    generation_messages = generation_messages[(-2 * generation_chunk_buffer_size):]

        if streamer is not None:
            return None, sr

        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)
        logger.info(f"[Request {request_id}] Stream generation completed, final shape: {concat_audio_out_ids.shape}")
        concat_wv = self._audio_tokenizer.decode(concat_audio_out_ids.unsqueeze(0))[0, 0]
        return concat_wv, sr


def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    # Mapping of Chinese punctuation to English punctuation
    chinese_to_english_punct = {
        "，": ", ",  # comma
        "。": ".",  # period
        "：": ":",  # colon
        "；": ";",  # semicolon
        "？": "?",  # question mark
        "！": "!",  # exclamation mark
        "（": "(",  # left parenthesis
        "）": ")",  # right parenthesis
        "【": "[",  # left square bracket
        "】": "]",  # right square bracket
        "《": "<",  # left angle quote
        "》": ">",  # right angle quote
        """: '"',  # left double quotation
        """: '"',  # right double quotation
        "'": "'",  # left single quotation
        "'": "'",  # right single quotation
        "、": ",",  # enumeration comma
        "—": "-",  # em dash
        "…": "...",  # ellipsis
        "·": ".",  # middle dot
        "「": '"',  # left corner bracket
        "」": '"',  # right corner bracket
        "『": '"',  # left double corner bracket
        "』": '"',  # right double corner bracket
    }

    # Replace each Chinese punctuation with its English counterpart
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)

    return text


def prepare_chunk_text(
    text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1
):
    """Chunk the text into smaller pieces. We will later feed the chunks one by one to the model.

    Parameters
    ----------
    text : str
        The text to be chunked.
    chunk_method : str, optional
        The method to use for chunking. Options are "speaker", "word", or None. By default, we won't use any chunking and
        will feed the whole text to the model.
    replace_speaker_tag_with_special_tags : bool, optional
        Whether to replace speaker tags with special tokens, by default False
        If the flag is set to True, we will replace [SPEAKER0] with <|speaker_id_start|>SPEAKER0<|speaker_id_end|>
    chunk_max_word_num : int, optional
        The maximum number of words for each chunk when "word" chunking method is used, by default 100
    chunk_max_num_turns : int, optional
        The maximum number of turns for each chunk when "speaker" chunking method is used,

    Returns
    -------
    List[str]
        The list of text chunks.

    """
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        # TODO: We may improve the logic in the future
        # For long-form generation, we will first divide the text into multiple paragraphs by splitting with "\n\n"
        # After that, we will chunk each paragraph based on word count
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            if language == "zh":
                # For Chinese, we will chunk based on character count
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def _build_system_message_with_audio_prompt(system_message):
    contents = []

    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN) :]

    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    ret = Message(
        role="system",
        content=contents,
    )
    return ret


def prepare_generation_context(scene_prompt, ref_audio, ref_audio_in_system_message, audio_tokenizer, speaker_tags):
    system_message, messages, audio_ids = None, [], []
    if ref_audio:
        speaker_info_l = ref_audio.split(",")
        voice_profile = None
        if any(s.startswith("profile:") for s in speaker_info_l): ref_audio_in_system_message = True
        if ref_audio_in_system_message:
            speaker_desc = []
            for spk_id, character_name in enumerate(speaker_info_l):
                if character_name.startswith("profile:"):
                    if not voice_profile:
                        with open(f"{CURR_DIR}/voice_prompts/profile.yaml", "r", encoding="utf-8") as f: voice_profile = yaml.safe_load(f)
                    character_desc = voice_profile["profiles"][character_name[len("profile:") :].strip()]
                    speaker_desc.append(f"SPEAKER{spk_id}: {character_desc}")
                else: speaker_desc.append(f"SPEAKER{spk_id}: {AUDIO_PLACEHOLDER_TOKEN}")
            system_message_content = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>" if scene_prompt else "Generate audio following instruction.\n\n<|scene_desc_start|>\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>"
            system_message = _build_system_message_with_audio_prompt(system_message_content)
        else:
            if scene_prompt: system_message = Message(role="system", content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
        for spk_id, character_name in enumerate(speaker_info_l):
            if not character_name.startswith("profile:"):
                prompt_audio_path, prompt_text_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{character_name}.wav"), os.path.join(f"{CURR_DIR}/voice_prompts", f"{character_name}.txt")
                assert os.path.exists(prompt_audio_path), (
                    f"Voice prompt audio file {prompt_audio_path} does not exist."
                )
                assert os.path.exists(prompt_text_path), f"Voice prompt text file {prompt_text_path} does not exist."
                with open(prompt_text_path, "r", encoding="utf-8") as f: prompt_text = f.read().strip()
                audio_ids.append(audio_tokenizer.encode(prompt_audio_path))
                if not ref_audio_in_system_message:
                    messages.extend([Message(role="user", content=f"[SPEAKER{spk_id}] {prompt_text}" if len(speaker_info_l) > 1 else prompt_text), Message(role="assistant", content=AudioContent(audio_url=prompt_audio_path))])
    else:
        if len(speaker_tags) > 1:
            speaker_desc = "\n".join([f"{tag}: {'feminine' if i % 2 == 0 else 'masculine'}" for i, tag in enumerate(speaker_tags)])
            scene_desc = f"{scene_prompt}\n\n{speaker_desc}" if scene_prompt else speaker_desc
            system_message = Message(role="system", content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>")
        else:
            system_message_content = "Generate audio following instruction."
            if scene_prompt: system_message_content += f"\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"
            system_message = Message(role="system", content=system_message_content)
    if system_message: messages.insert(0, system_message)
    return messages, audio_ids


def prepare_multi_emotion_context(emotions: List[EmotionConfig], audio_tokenizer, scene_prompt: Optional[str] = None):
    """准备多情感生成上下文"""
    logger.info(f"=== 开始准备多情感上下文 ===")
    logger.info(f"情感配置数量: {len(emotions)}")
    logger.info(f"场景描述: {scene_prompt}")

    messages = []
    audio_ids = []

    # 构建情感场景描述
    emotion_descriptions = []
    for i, emotion_config in enumerate(emotions):
        emotion_desc = f"EMOTION_{i}: {emotion_config.emotion} (intensity: {emotion_config.intensity})"
        if emotion_config.scene_prompt:
            emotion_desc += f" - {emotion_config.scene_prompt}"
        emotion_descriptions.append(emotion_desc)
        logger.info(f"情感描述 {i}: {emotion_desc}")

    # 构建系统消息 - 让模型理解如何利用参考音频
    system_content = "You are an advanced audio generation model with multiple emotional styles.\n\n"
    if scene_prompt:
        system_content += f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>\n\n"
        logger.info(f"添加场景描述到系统消息: {scene_prompt}")

    system_content += "Available emotional styles:\n" + "\n".join(emotion_descriptions) + "\n\n"
    system_content += "Instructions:\n"
    system_content += "1. Analyze the input text to understand its emotional content and context\n"
    system_content += "2. Choose the most appropriate emotional style from the available options\n"
    system_content += "3. Use the corresponding reference audio as a style guide\n"
    system_content += "4. Apply the emotional intensity level when generating audio\n"
    system_content += "5. Maintain natural and coherent emotional expression throughout the audio\n"
    system_content += "6. You can blend multiple emotional styles if the content requires it\n"

    logger.info("系统消息已构建")
    logger.info(f"系统消息长度: {len(system_content)} 字符")

    system_message = Message(role="system", content=system_content)
    messages.append(system_message)
    logger.info("系统消息已添加到消息列表")

    # 为每个情感添加参考音频示例
    logger.info("开始处理参考音频...")
    for i, emotion_config in enumerate(emotions):
        prompt_audio_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{emotion_config.ref_audio}.wav")
        prompt_text_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{emotion_config.ref_audio}.txt")

        logger.info(f"处理情感 {i}: {emotion_config.emotion}")
        logger.info(f"  音频文件路径: {prompt_audio_path}")
        logger.info(f"  文本文件路径: {prompt_text_path}")

        assert os.path.exists(prompt_audio_path), f"Voice prompt audio file {prompt_audio_path} does not exist."
        assert os.path.exists(prompt_text_path), f"Voice prompt text file {prompt_text_path} does not exist."

        logger.info("  文件存在性检查通过")

        with open(prompt_text_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()

        logger.info(f"  参考文本: {prompt_text}")

        # 添加情感标签到示例文本，让模型学习情感标签的使用
        example_text = f"[EMOTION_{i}:{emotion_config.emotion}:{emotion_config.intensity}] {prompt_text}"
        logger.info(f"  示例文本: {example_text}")

        # 编码音频
        audio_encoded = audio_tokenizer.encode(prompt_audio_path)
        audio_ids.append(audio_encoded)
        logger.info(f"  音频编码完成，形状: {audio_encoded.shape}")

        messages.extend([
            Message(role="user", content=example_text),
            Message(role="assistant", content=AudioContent(audio_url=prompt_audio_path))
        ])
        logger.info(f"  消息已添加到列表")

    logger.info(f"=== 多情感上下文准备完成 ===")
    logger.info(f"消息数量: {len(messages)}")
    logger.info(f"音频ID数量: {len(audio_ids)}")

    return messages, audio_ids


def prepare_multi_emotion_text_with_segments(text: str, emotions: List[EmotionConfig]) -> str:
    """为多情感生成准备带情感标签的文本"""
    logger.info(f"=== 开始多情感文本处理 ===")
    logger.info(f"输入文本: {text}")
    logger.info(f"情感配置数量: {len(emotions) if emotions else 0}")

    if not emotions:
        logger.info("没有配置情感，返回原文本")
        return text

    # 打印情感配置详情
    for i, emotion_config in enumerate(emotions):
        logger.info(f"情感 {i}: {emotion_config.emotion} (强度: {emotion_config.intensity}, 参考音频: {emotion_config.ref_audio})")
        if emotion_config.scene_prompt:
            logger.info(f"  场景描述: {emotion_config.scene_prompt}")

    # 方案1: 让模型自动选择情感（推荐）
    def prepare_for_auto_selection(text: str, emotions: List[EmotionConfig]) -> str:
        """让大模型根据文本内容自动选择合适的情感"""
        logger.info("使用自动情感选择方案")

        # 构建情感选择指令
        emotion_descriptions = []
        for i, emotion_config in enumerate(emotions):
            desc = f"EMOTION_{i}: {emotion_config.emotion}"
            if emotion_config.scene_prompt:
                desc += f" ({emotion_config.scene_prompt})"
            emotion_descriptions.append(desc)

        logger.info(f"可用情感描述: {emotion_descriptions}")

        # 创建选择指令
        selection_instruction = f"""
请根据以下文本内容，从���用的情感中选择最合适的一个来表达：

可用情感：
{chr(10).join(emotion_descriptions)}

文本内容：{text}

请用选定的情感标签标记文本：[EMOTION_X:情感名称:强度]
"""

        logger.info("选择指令已构建")
        logger.info("直接返回原文本，让模型通过上下文学习")

        # 简化版本：直接返回原文本，让模型通过上下文学习
        return text

    # 方案2: 均匀分配（用于测试不同情感效果）
    def assign_emotions_evenly(text: str, emotions: List[EmotionConfig]) -> str:
        """均匀分配情感到文本段落，用于测试"""
        logger.info("使用均匀分配方案（测试模式）")

        sentences = re.split(r'[.!?]+', text.strip())
        logger.info(f"文本分割为 {len(sentences)} 个句子")

        processed_sentences = []

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # 循环使用情感配置
            emotion_index = i % len(emotions)
            emotion_config = emotions[emotion_index]
            processed_sentence = f"[EMOTION_{emotion_index}:{emotion_config.emotion}:{emotion_config.intensity}] {sentence}"
            processed_sentences.append(processed_sentence)

            logger.info(f"句子 {i+1}: {sentence}")
            logger.info(f"  分配情感: {emotion_config.emotion} (索引: {emotion_index})")
            logger.info(f"  处理后: {processed_sentence}")

        result = ". ".join(processed_sentences) + "."
        logger.info(f"均匀分配完成，最终结果: {result}")
        return result

    # 使用自动选择方案
    logger.info("选择使用自动情感选择方案")
    result = prepare_for_auto_selection(text, emotions)

    logger.info(f"=== 多情感文本处理完成 ===")
    logger.info(f"���终输出: {result}")
    logger.info(f"输出长度: {len(result)} 字符")

    return result





# FastAPI App
app = FastAPI()
model_client: Optional[HiggsAudioModelClient] = None
audio_tokenizer_global = None

# 添加请求跟踪
class RequestTracker:
    def __init__(self):
        self._active_requests = {}
        self._lock = threading.Lock()
    
    def start_request(self, request_id: str):
        with self._lock:
            self._active_requests[request_id] = time.time()
            logger.info(f"Active requests: {len(self._active_requests)} ({list(self._active_requests.keys())})")
    
    def end_request(self, request_id: str):
        with self._lock:
            if request_id in self._active_requests:
                duration = time.time() - self._active_requests[request_id]
                del self._active_requests[request_id]
                logger.info(f"Request {request_id} completed in {duration:.2f}s. Active requests: {len(self._active_requests)}")

request_tracker = RequestTracker()


class GenerationRequest(BaseModel):
    transcript: str = Field(..., example="Hello, this is a test.", description="Text to be converted to speech.")
    ref_audio: Optional[str] = Field(None, example="broom_salesman", description="Reference audio name for voice cloning.")
    scene_prompt: Optional[str] = Field(None, description="Scene description prompt.")

    # 多情感支持
    emotions: Optional[List[EmotionConfig]] = Field(None, description="多情感配置列表")

    temperature: float = Field(1.0, description="Sampling temperature.")
    top_k: int = Field(50, description="Top-k filtering.")
    top_p: float = Field(0.95, description="Top-p (nucleus) filtering.")
    ras_win_len: int = Field(7, description="RAS window length.")
    ras_win_max_num_repeat: int = Field(2, description="RAS window max repeat.")
    ref_audio_in_system_message: bool = Field(False, description="Include ref audio desc in system message.")
    chunk_method: Optional[str] = Field(None, example="word", description="Chunking method: 'speaker', 'word', or None.")
    chunk_max_word_num: int = Field(200, description="Max words per chunk for 'word' method.")
    chunk_max_num_turns: int = Field(1, description="Max turns per chunk for 'speaker' method.")
    generation_chunk_buffer_size: Optional[int] = Field(None, description="Buffer size for generated audio chunks.")
    seed: Optional[int] = Field(12345, description="Random seed for generation.")
    max_new_tokens: int = Field(1024, description="Maximum new tokens to generate.")
    sample_rate: int = Field(32000, description="Sample rate for generated audio.")

class AudioSpeechRequest(BaseModel):
    model: str = "higgs-audio-v2-generation-3B-base"
    """ The model to use for the audio speech request. """

    input: str
    """ The input to the audio speech request. """

    voice: str
    """ The voice to use for the audio speech request. """

    speed: float = 1.0
    """ The speed of the audio speech request. """

    temperature: float = 1.0
    """ The temperature of the audio speech request. """

    top_p: float = 0.95
    """ The top p of the audio speech request. """

    top_k: int = 50
    """ The top k of the audio speech request. """

    response_format: Literal["wav", "mp3", "pcm"] = "pcm"
    """ The response format of the audio speech request. """

    stop: Optional[list[str]] = None

    max_tokens: int = 1024

    sample_rate: int = 24000
    """ The sample rate of the audio speech request. """
    
    seed: Optional[int] = None
    """ The seed for random number generation. """
    
    ras_win_len: int = 7
    """ The window length for RAS. """
    ras_win_max_num_repeat: int = 2
    """ The maximum number of repeats for RAS. """
    generation_chunk_buffer_size: Optional[int] = None
    """ Buffer size for generated audio chunks. """


@app.on_event("startup")
def load_model():
    global model_client, audio_tokenizer_global
    logger.info("Loading model and tokenizer...")
    model_path = "/data6/chen.yuxiang/models/higgs-audio-v2-generation-3B-base"
    audio_tokenizer_path = "/data6/chen.yuxiang/models/higgs-audio-v2-tokenizer"
    use_static_kv_cache = torch.cuda.is_available()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    audio_tokenizer_global = load_higgs_audio_tokenizer(audio_tokenizer_path, device=device)
    logger.info(f"Audio tokenizer loaded with sample rate: {audio_tokenizer_global.sampling_rate}")

    model_client = HiggsAudioModelClient(
        model_path=model_path,
        audio_tokenizer=audio_tokenizer_global,
        device_id=0 if torch.cuda.is_available() else None,
        use_static_kv_cache=use_static_kv_cache,
    )
    logger.info("Model and tokenizer loaded successfully.")


@app.post("/v1/audio/speech")
async def create_audio_speech(request: AudioSpeechRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    if not model_client:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    request_tracker.start_request(request_id)
    
    try:
        logger.info(f"[Request {request_id}] Starting audio generation")
        
        transcript = request.input
        pattern = re.compile(r"\[(SPEAKER\d+)\]")
        speaker_tags = sorted(set(pattern.findall(transcript)))

        transcript = normalize_chinese_punctuation(transcript)
        transcript = transcript.replace("(", " ").replace(")", " ")
        transcript = transcript.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")

        replacements = {"[laugh]": "<SE>[Laughter]</SE>", "[humming start]": "<SE>[Humming]</SE>", "[humming end]": "<SE_e>[Humming]</SE_e>", "[music start]": "<SE_s>[Music]</SE_s>", "[music end]": "<SE_e>[Music]</SE_e>", "[music]": "<SE>[Music]</SE>", "[sing start]": "<SE_s>[Singing]</SE_s>", "[sing end]": "<SE_e>[Singing]</SE_e>", "[applause]": "<SE>[Applause]</SE>", "[cheering]": "<SE>[Cheering]</SE>", "[cough]": "<SE>[Cough]</SE>"}
        for tag, rep in replacements.items():
            transcript = transcript.replace(tag, rep)

        transcript = "\n".join([" ".join(line.split()) for line in transcript.split("\n") if line.strip()]).strip()
        if not any(transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]):
            transcript += "."


        messages, audio_ids = prepare_generation_context(
            scene_prompt="generate voice from text: " + transcript,
            ref_audio=request.voice,
            ref_audio_in_system_message=False,
            audio_tokenizer=audio_tokenizer_global,
            speaker_tags=speaker_tags,
        )
        processed_transcript = transcript

        chunked_text = prepare_chunk_text(
            processed_transcript,
            chunk_method=None,
        )

        concat_wv, sr = model_client.generate(
            messages=messages,
            audio_ids=audio_ids,
            chunked_text=chunked_text,
            generation_chunk_buffer_size=request.generation_chunk_buffer_size,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            ras_win_len=request.ras_win_len,
            ras_win_max_num_repeat=request.ras_win_max_num_repeat,
            seed=request.seed,
            max_new_tokens=request.max_tokens,
            request_id=request_id,  # 传递请求ID
        )

        # Save to in-memory file
        buffer = io.BytesIO()
        if isinstance(concat_wv, torch.Tensor):
            audio_np = concat_wv.cpu().numpy()
        else:
            audio_np = concat_wv  # already numpy array

        # 使用音频tokenizer的实际采样率
        actual_sample_rate = model_client._audio_tokenizer.sampling_rate
        remove_leading_silence(audio_np, sample_rate=actual_sample_rate)
        if request.sample_rate != actual_sample_rate:
            audio_np = librosa.resample(audio_np, orig_sr=actual_sample_rate, target_sr=request.sample_rate)
        logger.info(f"[Request {request_id}] Audio shape after resampling: {audio_np.shape}, sample rate: {request.sample_rate}")
        sf.write(buffer, audio_np, request.sample_rate, format='WAV')
        buffer.seek(0)

        elapsed = time.time() - start_time
        logger.info(f"[Request {request_id}] Request processed in {elapsed:.2f} seconds")
        
        request_tracker.end_request(request_id)
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Request {request_id}] Error during audio generation after {elapsed:.2f}s: {e}")
        request_tracker.end_request(request_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_audio(request: GenerationRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    if not model_client:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    request_tracker.start_request(request_id)
    
    try:
        logger.info(f"[Request {request_id}] Starting audio generation")
        
        transcript = request.transcript
        pattern = re.compile(r"\[(SPEAKER\d+)\]")
        speaker_tags = sorted(set(pattern.findall(transcript)))

        transcript = normalize_chinese_punctuation(transcript)
        transcript = transcript.replace("(", " ").replace(")", " ")
        transcript = transcript.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")

        replacements = {"[laugh]": "<SE>[Laughter]</SE>", "[humming start]": "<SE>[Humming]</SE>", "[humming end]": "<SE_e>[Humming]</SE_e>", "[music start]": "<SE_s>[Music]</SE_s>", "[music end]": "<SE_e>[Music]</SE_e>", "[music]": "<SE>[Music]</SE>", "[sing start]": "<SE_s>[Singing]</SE_s>", "[sing end]": "<SE_e>[Singing]</SE_e>", "[applause]": "<SE>[Applause]</SE>", "[cheering]": "<SE>[Cheering]</SE>", "[cough]": "<SE>[Cough]</SE>"}
        for tag, rep in replacements.items():
            transcript = transcript.replace(tag, rep)

        transcript = "\n".join([" ".join(line.split()) for line in transcript.split("\n") if line.strip()]).strip()
        if not any(transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]):
            transcript += "."

        # 检查是否使用多情感模式
        if request.emotions:
            messages, audio_ids = prepare_multi_emotion_context(
                emotions=request.emotions,
                audio_tokenizer=audio_tokenizer_global,
                scene_prompt=request.scene_prompt,
            )

            # 为多情感模式准备带情感标签的文本
            processed_transcript = prepare_multi_emotion_text_with_segments(
                transcript,
                request.emotions
            )
        else:
            messages, audio_ids = prepare_generation_context(
                scene_prompt=request.scene_prompt,
                ref_audio=request.ref_audio,
                ref_audio_in_system_message=request.ref_audio_in_system_message,
                audio_tokenizer=audio_tokenizer_global,
                speaker_tags=speaker_tags,
            )
            processed_transcript = transcript

        chunked_text = prepare_chunk_text(
            processed_transcript,
            chunk_method=request.chunk_method,
            chunk_max_word_num=request.chunk_max_word_num,
            chunk_max_num_turns=request.chunk_max_num_turns,
        )

        concat_wv, sr = model_client.generate(
            messages=messages,
            audio_ids=audio_ids,
            chunked_text=chunked_text,
            generation_chunk_buffer_size=request.generation_chunk_buffer_size,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            ras_win_len=request.ras_win_len,
            ras_win_max_num_repeat=request.ras_win_max_num_repeat,
            seed=request.seed,
            max_new_tokens=request.max_new_tokens,
            request_id=request_id,  # 传递请求ID
        )

        # Save to in-memory file
        buffer = io.BytesIO()
        if isinstance(concat_wv, torch.Tensor):
            audio_np = concat_wv.cpu().numpy()
        else:
            audio_np = concat_wv  # already numpy array

        # 使用音频tokenizer的实际采样率
        actual_sample_rate = model_client._audio_tokenizer.sampling_rate
        remove_leading_silence(audio_np, sample_rate=actual_sample_rate)
        if request.sample_rate != actual_sample_rate:
            audio_np = librosa.resample(audio_np, orig_sr=actual_sample_rate, target_sr=request.sample_rate)
        logger.info(f"[Request {request_id}] Audio shape after resampling: {audio_np.shape}, sample rate: {request.sample_rate}")
        sf.write(buffer, audio_np, request.sample_rate, format='WAV')
        buffer.seek(0)

        elapsed = time.time() - start_time
        logger.info(f"[Request {request_id}] Request processed in {elapsed:.2f} seconds")
        
        request_tracker.end_request(request_id)
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Request {request_id}] Error during audio generation after {elapsed:.2f}s: {e}")
        request_tracker.end_request(request_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/streaming")
async def stream_generate_audio(request: GenerationRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    if not model_client:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    request_tracker.start_request(request_id)
    
    try:
        logger.info(f"[Request {request_id}] Starting streaming audio generation")
        
        return StreamingResponse(
            stream_generate(request, request_id),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Request {request_id}] Error during streaming generation after {elapsed:.2f}s: {e}")
        request_tracker.end_request(request_id)
        raise HTTPException(status_code=500, detail=str(e))


def remove_leading_silence(pcm_data, sample_rate=44100, silence_threshold=0.008, min_silence_duration=0.1, keep_leading_silence=0.05):
    """
    去除前置静音片段，���保留指定长度的前置静音
    参数:
        pcm_data: numpy数组，float格式，PCM音频数据
    返回:
        处理后的PCM数据
    """

    # 计算帧数和各种时间对应的帧数
    num_samples = len(pcm_data)
    min_silence_frames = int(min_silence_duration * sample_rate)
    keep_frames = int(keep_leading_silence * sample_rate)

    # 计算短时能量(绝对值)用于静音检测
    window_size = int(0.05 * sample_rate)  # 50ms窗口
    energy = np.convolve(np.abs(pcm_data), np.ones(window_size)/window_size, mode='same')

    # 找到第一个超过阈值的点
    speech_start = 0
    for i in range(len(energy) - min_silence_frames):
        # 检查接下来的min_silence_frames帧是否都低于阈值
        if np.all(energy[i:i+min_silence_frames] < silence_threshold):
            speech_start = i + min_silence_frames
        else:
            break

    # 确保保留keep_frames的前置静音
    if speech_start > keep_frames:
        speech_start = max(0, speech_start - keep_frames)
    else:
        speech_start = 0

    return pcm_data[speech_start: ]

def crossfade_audio_chunks(pre_audio, cur_audio, fade_duration=0.01, sample_rate=24000):
    cross_fade_samples = int(fade_duration * sample_rate)
    if len(pre_audio) >= cross_fade_samples and len(cur_audio) >= cross_fade_samples:

        # 交叉淡化
        fade_out = np.linspace(1, 0, cross_fade_samples, dtype=np.float32)
        fade_in = np.linspace(0, 1, cross_fade_samples, dtype=np.float32)  
        # 混合重叠部分
        overlap = (pre_audio[-cross_fade_samples:] * fade_out +
                   cur_audio[:cross_fade_samples] * fade_in)
        pre_audio = np.concatenate([pre_audio[:-cross_fade_samples], overlap])
        cur_audio = cur_audio[cross_fade_samples:]

        return pre_audio, cur_audio
    else:
        return pre_audio, cur_audio


async def stream_generate(request: GenerationRequest, request_id: str):
    start = time.time()

    try:
        # 数据预处理
        transcript = request.transcript
        pattern = re.compile(r"\[(SPEAKER\d+)\]")
        speaker_tags = sorted(set(pattern.findall(transcript)))

        transcript = normalize_chinese_punctuation(transcript)
        transcript = transcript.replace("(", " ").replace(")", " ")
        transcript = transcript.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")

        replacements = {"[laugh]": "<SE>[Laughter]</SE>", "[humming start]": "<SE>[Humming]</SE>", "[humming end]": "<SE_e>[Humming]</SE_e>", "[music start]": "<SE_s>[Music]</SE_s>", "[music end]": "<SE_e>[Music]</SE_e>", "[music]": "<SE>[Music]</SE>", "[sing start]": "<SE_s>[Singing]</SE_s>", "[sing end]": "<SE_e>[Singing]</SE_e>", "[applause]": "<SE>[Applause]</SE>", "[cheering]": "<SE>[Cheering]</SE>", "[cough]": "<SE>[Cough]</SE>"}
        for tag, rep in replacements.items():
            transcript = transcript.replace(tag, rep)

        transcript = "\n".join([" ".join(line.split()) for line in transcript.split("\n") if line.strip()]).strip()
        if not any(transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]):
            transcript += "."

        # 检查是否使用多情感模式
        if request.emotions:
            logger.info(f"[Request {request_id}] === 检测到多情感模式 ===")
            logger.info(f"[Request {request_id}] 情感配置: {[e.emotion for e in request.emotions]}")

            messages, audio_ids = prepare_multi_emotion_context(
                emotions=request.emotions,
                audio_tokenizer=audio_tokenizer_global,
                scene_prompt=request.scene_prompt,
            )

            # 为多情感模式准备带情感标签的文本
            processed_transcript = prepare_multi_emotion_text_with_segments(
                transcript,
                request.emotions
            )

            logger.info(f"[Request {request_id}] 多情感处理完成")
            logger.info(f"[Request {request_id}] 原始文本: {transcript}")
            logger.info(f"[Request {request_id}] 处理后文本: {processed_transcript}")
        else:
            logger.info(f"[Request {request_id}] === 使用单情感模式 ===")
            messages, audio_ids = prepare_generation_context(
                scene_prompt=request.scene_prompt,
                ref_audio=request.ref_audio,
                ref_audio_in_system_message=request.ref_audio_in_system_message,
                audio_tokenizer=audio_tokenizer_global,
                speaker_tags=speaker_tags,
            )
            processed_transcript = transcript
            logger.info(f"[Request {request_id}] 单情感处理完成，使用原始文本")

        chunked_text = prepare_chunk_text(
            processed_transcript,
            chunk_method=request.chunk_method,
            chunk_max_word_num=request.chunk_max_word_num,
            chunk_max_num_turns=request.chunk_max_num_turns,
        )

        # 在模型推理部分使用 torch.inference_mode
        with torch.inference_mode():
            async_streamer = AsyncHiggsAudioStreamer(
                tokenizer=model_client._tokenizer,
                skip_prompt=True,
                audio_num_codebooks=8
            )

            generation_kwargs = dict(
                messages=messages,
                audio_ids=audio_ids,
                chunked_text=chunked_text,
                generation_chunk_buffer_size=request.generation_chunk_buffer_size,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                ras_win_len=request.ras_win_len,
                ras_win_max_num_repeat=request.ras_win_max_num_repeat,
                seed=request.seed,
                max_new_tokens=request.max_new_tokens,
                streamer=async_streamer,
                request_id=request_id,  # 传递请求ID
            )

            # 启动生成线程
            thread = threading.Thread(target=model_client.stream_generate, kwargs=generation_kwargs)
            thread.start()

            # 流式处理输出 - 收集所有音频token
            audio_out_ids_l = []
            overlap_data = None
            first_audio = 0
            pre_audio = None
            async for delta in async_streamer:
                if delta.audio_tokens is not None:
                    logger.debug(f"[Request {request_id}] Received audio tokens, shape: {delta.audio_tokens.shape}")

                    audio_tokens = delta.audio_tokens.unsqueeze(1)
                    audio_tokens = audio_tokens.to(model_client._audio_tokenizer.device)
                    audio_out_ids_l.append(audio_tokens)

                    if len(audio_out_ids_l) >= 9:
                        first_audio += 1
                        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)

                        if model_client._config.use_delay_pattern:
                            concat_audio_out_ids, overlap_data = streaming_revert_delay_pattern(concat_audio_out_ids, overlap_data)
                        
                        concat_audio_out_ids = concat_audio_out_ids.clip(0, model_client._audio_tokenizer.codebook_size - 1)
                        if first_audio == 1:
                            concat_audio_out_ids = concat_audio_out_ids[:, 1:]

                        wv = model_client._audio_tokenizer.decode(concat_audio_out_ids.unsqueeze(0))[0, 0]

                        if isinstance(wv, torch.Tensor):
                            wv = wv.detach().cpu().numpy()

                        actual_sample_rate = model_client._audio_tokenizer.sampling_rate

                        # 重采样到目标采样率
                        if request.sample_rate != actual_sample_rate:
                            wv = librosa.resample(wv, orig_sr=actual_sample_rate, target_sr=request.sample_rate)

                        if pre_audio is None:
                            pre_audio = wv
                            audio_out_ids_l = []
                        else:
                            pre_audio, wv = crossfade_audio_chunks(pre_audio, wv, fade_duration=0.04, sample_rate=request.sample_rate)
                            audio_int16 = (pre_audio * 32767).astype(np.int16)
                            audio_bytes = audio_int16.tobytes()
                            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                            response_data = {
                                "choices": [
                                    {
                                        "finish_reason": None,
                                        "delta": {
                                            "role": "assistant",
                                            "audio": {
                                                "data": audio_b64
                                            }
                                        },
                                        "index": 0
                                    }
                                ]
                            }

                            pre_audio = wv
                            audio_out_ids_l = []
                            yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"

            # 等待所有音频token收集完成后，只输出一次完整的音频
            if len(audio_out_ids_l) > 0:
                first_audio += 1
                logger.info(f"[Request {request_id}] All audio tokens collected, total chunks: {len(audio_out_ids_l)}")
                concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)

                if model_client._config.use_delay_pattern:
                    concat_audio_out_ids, overlap_data = streaming_revert_delay_pattern(concat_audio_out_ids, overlap_data)

                # 处理音频token的裁剪
                if first_audio == 1:
                    concat_audio_out_ids = concat_audio_out_ids.clip(0, model_client._audio_tokenizer.codebook_size - 1)[:, 1:-1]
                else:
                    concat_audio_out_ids = concat_audio_out_ids.clip(0, model_client._audio_tokenizer.codebook_size - 1)[:, 1:-1]

                # 解码音频
                if concat_audio_out_ids.shape[1] == 0:
                    logger.warning(f"[Request {request_id}] No audio tokens to decode, skipping output.")
                else:
                    wv = model_client._audio_tokenizer.decode(concat_audio_out_ids.unsqueeze(0))[0, 0]

                    if isinstance(wv, torch.Tensor):
                        wv = wv.detach().cpu().numpy()

                    actual_sample_rate = model_client._audio_tokenizer.sampling_rate

                    # 重采样到目标采样率
                    if request.sample_rate != actual_sample_rate:
                        wv = librosa.resample(wv, orig_sr=actual_sample_rate, target_sr=request.sample_rate)

                    if pre_audio is not None:
                        pre_audio, wv = crossfade_audio_chunks(pre_audio, wv, fade_duration=0.04, sample_rate=request.sample_rate)
                        wv = np.concatenate([pre_audio, wv], axis=0)

                    # 转换为int16格式并编码为base64
                    audio_int16 = (wv * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                    # 输出完整的音频数据
                    response_data = {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "delta": {
                                    "role": "assistant",
                                    "audio": {
                                        "data": audio_b64
                                    }
                                },
                                "index": 0
                            }
                        ]
                    }

                    yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"

            # 等待线程完成
            thread.join()
            
            elapsed = time.time() - start
            logger.info(f"[Request {request_id}] Streaming generation completed in {elapsed:.2f} seconds")
            request_tracker.end_request(request_id)
            
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[Request {request_id}] Error in streaming generation after {elapsed:.2f}s: {e}")
        request_tracker.end_request(request_id)
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6001)