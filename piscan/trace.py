"""Context segments and execution traces (Sec. 2.2 definitions).

Every piece of text that reaches the model is a Segment with a source:
  sys  - system prompt            (trusted)
  user - user query               (trusted)
  retr - retrieved content        (untrusted)
  tool - tool output              (untrusted)
A Trace records the segments seen, the tool calls made and the final answer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

TRUSTED = {"sys", "user"}
UNTRUSTED = {"retr", "tool"}


@dataclass
class Segment:
    source: str                  # sys | user | retr | tool
    text: str
    carrier: str = ""            # e.g. "rag_doc:3", "email:2", "fetch_url:http://..."
    injected: bool = False       # ground truth: did the scanner place a payload here?

    @property
    def trusted(self) -> bool:
        return self.source in TRUSTED


@dataclass
class ToolCall:
    step: int
    name: str
    arguments: dict
    result: str = ""


@dataclass
class Trace:
    app: str
    task_id: str
    segments: list[Segment] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    final: str = ""
    raw_outputs: list[str] = field(default_factory=list)
    steps: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
