"""Model backends.

A backend turns a list of chat messages (and optional tool schemas) into the
raw text the model generates. Tool calls are parsed later by the app, so the
backend stays model-agnostic.

HFBackend   - local Hugging Face model, greedy decoding (Kaggle / Colab GPU).
MockBackend - scripted responses for CPU tests, no model needed.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional


class Backend:
    name = "base"

    def generate(self, messages: list[dict], tools: Optional[list[dict]] = None) -> str:
        raise NotImplementedError


class HFBackend(Backend):
    """Greedy decoding with a chat template that supports tools (Qwen2.5, Llama 3.1)."""

    def __init__(self, model_id: str, max_new_tokens: int = 384, dtype: str = "float16",
                 load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self._torch = torch

    def generate(self, messages, tools=None):
        prompt = self.tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tok.eos_token_id)
        new = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True)


class MockBackend(Backend):
    """Returns responses from a user-supplied policy function.

    The policy sees the messages and returns a string. Used to test the
    pipeline without a GPU; it is not a model of real behaviour.
    """

    def __init__(self, policy: Callable[[list[dict], Optional[list[dict]]], str], name="mock"):
        self.policy = policy
        self.name = name

    def generate(self, messages, tools=None):
        return self.policy(messages, tools)


# ---------------------------------------------------------------- tool-call parsing

_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract Qwen/Hermes-style <tool_call>{json}</tool_call> blocks.

    Malformed JSON is skipped rather than raised: a model that emits broken
    tool syntax has not made a tool call.
    """
    calls = []
    for m in _TOOL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if name:
            calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return calls


def strip_tool_calls(text: str) -> str:
    return _TOOL_RE.sub("", text).strip()
