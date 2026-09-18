from __future__ import annotations
import hashlib
import os
import torch
import torch.nn as nn


def qwen3vl_fingerprint(
    model_path: str, *, chat_template: bool, add_gen: bool, layer: int
) -> str:
    st = os.stat(os.path.join(model_path, "model.safetensors"))
    raw = f"{st.st_size}|chat={int(chat_template)}|addgen={int(add_gen)}|layer={layer}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class Qwen3VLTextEncoder(nn.Module):
    def __init__(
        self,
        model_path: str,
        device: torch.device | str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        chat_template: bool = True,
        add_generation_prompt: bool = True,
        layer: int = -1,
        max_length: int = 512,
    ):
        super().__init__()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self.model_path = model_path
        self.device = torch.device(device)
        self.chat_template = bool(chat_template)
        self.add_generation_prompt = bool(add_generation_prompt)
        self.layer = int(layer)
        self.max_length = int(max_length)
        from pathlib import Path

        hf_home = Path(
            os.environ.setdefault("HF_HOME", "/working/huggingface_cache")
        ).resolve()
        if not Path(model_path).resolve().is_relative_to(hf_home / "hub"):
            raise ValueError(
                "The text encoder must be a local snapshot beneath HF_HOME/hub."
            )
        if not Path(model_path).is_dir():
            raise FileNotFoundError(
                "Local text encoder snapshot not found; no automatic downloads are performed."
            )
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_path, dtype=dtype, local_files_only=True
            )
            .to(self.device)
            .eval()
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size = int(self.model.config.text_config.hidden_size)

    @property
    def fingerprint(self) -> str:
        return qwen3vl_fingerprint(
            self.model_path,
            chat_template=self.chat_template,
            add_gen=self.add_generation_prompt,
            layer=self.layer,
        )

    def _build_inputs(self, texts: list[str]):
        if self.chat_template:
            convs = [
                [{"role": "user", "content": [{"type": "text", "text": t}]}]
                for t in texts
            ]
            inputs = self.processor.apply_chat_template(
                convs,
                add_generation_prompt=self.add_generation_prompt,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        else:
            inputs = self.processor.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._build_inputs(list(texts))
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[self.layer]
        mask = inputs["attention_mask"]
        return (hidden, mask)

    @torch.no_grad()
    def encode_valid_list(self, texts: list[str]) -> list[torch.Tensor]:
        hidden, mask = self.encode(texts)
        out = []
        for i in range(hidden.shape[0]):
            valid = mask[i].bool()
            out.append(hidden[i][valid].float().cpu())
        return out
