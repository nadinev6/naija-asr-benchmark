"""Audio loading and Whisper inference helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def load_audio_mono(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    array, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sr != target_sr:
        new_len = int(len(array) * target_sr / sr)
        array = np.interp(np.linspace(0, len(array) - 1, new_len), np.arange(len(array)), array).astype(np.float32)
    return array

def whisper_inputs_to_device(batch: dict[str, torch.Tensor], device: str, use_bf16: bool) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        t = v.to(device)
        if t.dtype == torch.float32:
            if use_bf16:
                t = t.to(torch.bfloat16)
            elif device == "cuda":
                t = t.to(torch.float16)
        out[k] = t
    return out

def load_whisper_model(model_dir: str | Path, device: str, hf_token: str | None = None) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, bool]:
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    if use_bf16:
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    import transformers.tokenization_utils_base as _tub
    _orig_sst = _tub.PreTrainedTokenizerBase._set_model_specific_special_tokens
    def _safe_sst(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {}
        return _orig_sst(self, special_tokens)
    _tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = _safe_sst
    load_kwargs = {}
    if hf_token:
        load_kwargs["token"] = hf_token
    try:
        processor = WhisperProcessor.from_pretrained(str(model_dir), **load_kwargs)
    finally:
        _tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = _orig_sst
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir), torch_dtype=dtype, **load_kwargs)
    model.to(device)
    model.eval()
    return model, processor, use_bf16

@torch.inference_mode()
def transcribe_one_with_confidence(model, processor, audio: np.ndarray, *, device: str, use_bf16: bool, language: str = "en", task: str = "transcribe", max_new_tokens: int = 225, num_beams: int = 10, no_repeat_ngram_size: int = 3) -> dict[str, Any]:
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = whisper_inputs_to_device(inputs, device, use_bf16)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, language=language, task=task, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, length_penalty=1.0, condition_on_prev_tokens=False, return_dict_in_generate=True, output_scores=True)
    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    if outputs.scores:
        all_logprobs = []
        for i, score in enumerate(outputs.scores):
            probs = torch.nn.functional.log_softmax(score, dim=-1)
            token_id = outputs.sequences[0, i + 1] if i + 1 < outputs.sequences.shape[1] else None
            if token_id is not None:
                all_logprobs.append(probs[0, token_id].item())
        avg_logprob = sum(all_logprobs) / len(all_logprobs) if all_logprobs else 0.0
    else:
        avg_logprob = 0.0
    return {"text": text, "avg_logprob": avg_logprob}