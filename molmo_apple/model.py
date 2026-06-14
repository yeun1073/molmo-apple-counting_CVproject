"""
Molmo 모델 로드 및 단일 이미지 pointing 추론 래퍼.

Molmo-7B-D-0924 기준 (1세대, 좌표 0~100 정규화).

load_mode:
  "4bit"  — 4비트 NF4 양자화, BitsAndBytes 사용 (~7GB VRAM)  [권장]
  "bf16"  — bfloat16, GPU 한도 초과분은 CPU 오프로드 (느림)
  "auto"  — device_map=auto, 전체 정밀도 (VRAM 24GB 이상 필요)
"""
from __future__ import annotations
from typing import List
from PIL import Image

from .parsing import Point, parse_points


class MolmoPointer:
    def __init__(self, model_id: str = "allenai/Molmo-7B-D-0924",
                 generation: int = 1, device: str = "cuda",
                 load_mode: str = "4bit", gpu_mem_gib: int = 11):
        self.model_id = model_id
        self.generation = generation
        self.device = device
        self.load_mode = load_mode
        self.gpu_mem_gib = gpu_mem_gib
        self.processor = None
        self.model = None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True,
            torch_dtype="auto", device_map="auto",
        )

        common = dict(trust_remote_code=True)

        if self.load_mode == "4bit":
            # 4비트 양자화: 약 6~7GB로 압축, 12GB GPU에 통째로 적재
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=bnb,
                device_map="auto", **common,
            )

        elif self.load_mode == "bf16":
            # bfloat16 + GPU 메모리 상한 -> 초과분 CPU로 오프로드 (느림)
            max_memory = {0: f"{self.gpu_mem_gib}GiB", "cpu": "48GiB"}
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16,
                device_map="auto", max_memory=max_memory, **common,
            )

        else:  # "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype="auto", device_map="auto", **common,
            )
        return self

    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 2000) -> str:
        import torch
        from transformers import GenerationConfig

        inputs = self.processor.process(images=[image], text=prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer,
            )

        generated_tokens = output[0, inputs["input_ids"].size(1):]
        text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text

    def point(self, image: Image.Image, prompt: str) -> List[Point]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        text = self._generate(image, prompt)
        return parse_points(text, image.width, image.height, self.generation)

    def raw_text(self, image: Image.Image, prompt: str) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self._generate(image, prompt)
