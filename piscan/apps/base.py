"""Target applications: the LLM-integrated systems under test.

Each app defines tasks (benign user requests), carriers (places where untrusted
content enters the context) and attacker goals with an automatic success check.
The scanner never edits the app; it only changes the untrusted data the app
reads, which matches the attacker model in Sec. 2.3.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..backends import Backend, parse_tool_calls, strip_tool_calls
from ..trace import Segment, ToolCall, Trace


@dataclass
class Task:
    id: str
    query: str
    carrier: str                    # where a payload is placed for this task
    meta: dict = field(default_factory=dict)


@dataclass
class Goal:
    id: str
    cls: str                        # action | exfil | text
    instruction: dict               # lang -> imperative clause, e.g. {"en": "...", "uz": "..."}
    check: Callable[[Trace], bool]  # True if the attacker's goal is met in the trace


@dataclass
class Injection:
    carrier: str
    payload: str
    position: str = "end"           # end | middle


def insert_payload(text: str, payload: str, position: str = "end") -> str:
    if position == "middle":
        parts = text.split("\n\n")
        mid = max(1, len(parts) // 2)
        return "\n\n".join(parts[:mid] + [payload] + parts[mid:])
    return text.rstrip() + "\n\n" + payload


class App:
    name = "base"
    system_prompt = ""
    tools: list[dict] = []
    tasks: list[Task] = []
    goals: list[Goal] = []
    max_steps = 4

    # -- to be implemented by tool-using apps
    def make_env(self, task: Task, injection: Optional[Injection]):
        return None

    def call_tool(self, env, name: str, args: dict) -> str:
        return f"Error: unknown tool '{name}'"

    def tool_segment(self, env, name: str, args: dict, result: str) -> Segment:
        return Segment("tool", result, carrier=f"{name}")

    # -- generic agent loop
    def run(self, backend: Backend, task: Task, injection: Optional[Injection] = None) -> Trace:
        env = self.make_env(task, injection)
        tr = Trace(app=self.name, task_id=task.id)
        tr.segments.append(Segment("sys", self.system_prompt))
        tr.segments.append(Segment("user", task.query))
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task.query}]
        seen_calls: set = set()
        for step in range(self.max_steps):
            tr.steps = step + 1
            try:
                out = backend.generate(messages, self.tools or None)
            except Exception as e:  # keep the run, record the failure
                tr.error = f"generate: {e!r}"
                break
            tr.raw_outputs.append(out)
            calls = parse_tool_calls(out)
            if not calls:
                tr.final = strip_tool_calls(out)
                break
            messages.append({"role": "assistant", "content": strip_tool_calls(out),
                             "tool_calls": [{"type": "function",
                                             "function": {"name": c["name"],
                                                          "arguments": c["arguments"]}}
                                            for c in calls]})
            sig = tuple(sorted((c["name"], json.dumps(c["arguments"], sort_keys=True))
                               for c in calls))
            repeated = sig in seen_calls
            seen_calls.add(sig)
            for c in calls:
                result = self.call_tool(env, c["name"], c["arguments"])
                tr.tool_calls.append(ToolCall(step, c["name"], c["arguments"], result))
                seg = self.tool_segment(env, c["name"], c["arguments"], result)
                seg.injected = bool(injection and injection.payload in result)
                tr.segments.append(seg)
                messages.append({"role": "tool", "name": c["name"], "content": result})
            if repeated:                     # model is looping on the same call
                tr.final = ""
                break
        else:
            tr.error = "max_steps"
        return tr
