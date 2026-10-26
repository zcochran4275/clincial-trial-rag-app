"""OpenAI settings and helpers shared by both agents (extraction loop and RAG).

Tool definitions stay as {name, description, input_schema}; `openai_tools` wraps them in the
Chat Completions function-calling format.
Docs: https://platform.openai.com/docs/guides/function-calling
"""
from __future__ import annotations

import json
import os
from typing import Any

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")


def client():
    # imported here so tests don't need the SDK; reads OPENAI_API_KEY
    from openai import OpenAI

    return OpenAI()


def openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in tools
    ]


def assistant_message(msg) -> dict[str, Any]:
    """Echo the model's turn back into the history, including any tool calls it made."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return out


def tool_args(tc) -> dict[str, Any]:
    """Arguments arrive as a JSON string; malformed input raises ValueError/TypeError for the caller to report."""
    args = json.loads(tc.function.arguments or "{}")
    if not isinstance(args, dict):
        raise TypeError("tool arguments must be a JSON object")
    return args


def tool_result(tc, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tc.id, "content": content}
