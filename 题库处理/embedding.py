"""Shared local image/text encoder and index compatibility checks."""
import json
import os
import sys
import time

import numpy as np

from common import ROOT

MODEL = ROOT / 'models/Qwen3-VL-Embedding-2B'
MAX_PIXELS = 1800 * 32 * 32
INSTRUCTION = "Represent the user's input."


def encoder_settings(max_pixels=MAX_PIXELS):
    if not 4096 <= max_pixels <= MAX_PIXELS:
        raise ValueError('max-pixels 必须在 4096 到 1843200 之间')
    manifest = json.loads((MODEL / 'download-manifest.json').read_text())
    checksum = next(f['Sha256'] for f in manifest if f['Path'] == 'model.safetensors')
    return json.dumps(dict(model=MODEL.name, sha256=checksum, max_pixels=max_pixels,
                           instruction=INSTRUCTION, pooling='last_nonpadding', dtype='float16'), sort_keys=True)


def check_encoder(db, max_pixels=MAX_PIXELS, initialize=False):
    settings = encoder_settings(max_pixels)
    old = db.execute("SELECT value FROM settings WHERE key='encoder'").fetchone()
    unit = db.execute("SELECT value FROM settings WHERE key='unit'").fetchone()
    if (old and old[0] != settings) or (not old and not initialize):
        raise ValueError('编码设置不同或缺失，请使用与建库一致的模型和参数')
    if (unit and unit != ('question',)) or (not unit and not initialize):
        raise ValueError('需要逐题向量索引，不能使用整页索引')
    if initialize:
        with db:
            db.execute("INSERT OR IGNORE INTO settings VALUES ('encoder',?)", (settings,))
            db.execute("INSERT OR IGNORE INTO settings VALUES ('unit','question')")


def unit_vector(value):
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("向量必须是一维有限数值")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("不能索引零向量")
    return vector / norm

class Encoder:
    def __init__(self, max_pixels):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoProcessor, Qwen3VLModel
        torch.set_num_threads(2)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            # Preserve the memory budget on shared compute nodes.
            torch.cuda.set_per_process_memory_fraction(0.25)
        started = time.perf_counter()
        self.model, info = Qwen3VLModel.from_pretrained(
            MODEL, local_files_only=True, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            attn_implementation="sdpa", output_loading_info=True)
        if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
            raise ValueError("Embedding 权重不匹配: " + str(info))
        self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, padding_side="right")
        self.processor.image_processor.size = {"shortest_edge": 4096, "longest_edge": max_pixels}
        self.load_seconds = round(time.perf_counter() - started, 3)
        print("EMBED_MODEL_READY", self.device, self.load_seconds, file=sys.stderr, flush=True)

    def encode(self, texts=None, images=None):
        texts = texts if texts is not None else [None] * len(images)
        images = images if images is not None else [None] * len(texts)
        if not texts or len(texts) != len(images):
            raise ValueError("编码输入数量不一致或为空")
        conversations, pictures = [], []
        for text, picture in zip(texts, images):
            content = []
            if picture is not None:
                content.append({"type": "image"})
                pictures.append(picture)
            if text:
                content.append({"type": "text", "text": text})
            if not content:
                raise ValueError("编码内容为空")
            conversations.append([
                {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
                {"role": "user", "content": content}])
        rendered = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=rendered, images=pictures or None, padding=True, return_tensors="pt")
        if inputs["input_ids"].shape[1] > 8192:
            raise ValueError("输入过长，未截断图片或文字")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**inputs, use_cache=False).last_hidden_state
            mask = inputs["attention_mask"]
            last = mask.shape[1] - mask.flip(1).argmax(1) - 1
            vectors = hidden[self.torch.arange(hidden.shape[0], device=hidden.device), last].float()
            vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=-1)
        return vectors.cpu().numpy()
