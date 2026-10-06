from .base import App, Goal, Injection, Task
from .email_agent import EmailAgent
from .rag_qa import RagQA
from .web_summarizer import WebSummarizer

APPS = {a.name: a for a in (RagQA(), EmailAgent(), WebSummarizer())}

__all__ = ["App", "Goal", "Injection", "Task", "APPS", "RagQA", "EmailAgent", "WebSummarizer"]
