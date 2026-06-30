"""
tool_use_strategy.py

Tool use / function calling strategy with webhook-based tool execution.

How it works:
    1. User registers tools via POST /v1/tools/register with:
           { name, description, parameters_schema, webhook_url }
    2. This strategy loads registered tools and passes them to the LLM
       using the provider's native function-calling API.
    3. If the LLM decides to call a tool, the engine POSTs the tool
       arguments to the webhook URL and feeds the result back.
    4. The LLM generates a final answer using the tool result.

Built-in tools (always available, no webhook needed):
    - get_current_datetime  → returns current UTC datetime
    - calculate             → evaluates a math expression safely

Supported providers for function calling:
    - OpenAI / Groq  (same API: tools parameter in chat completions)
    - Gemini         (tools parameter with FunctionDeclaration)
    - Mock           (always returns "tool not supported" gracefully)

Webhook contract:
    POST {webhook_url}
    Body: {"tool_name": "...", "arguments": {...}}
    Expected response: {"result": <any JSON-serializable value>}
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from strategies.base_strategy import PromptStrategy

_DB_PATH = Path(__file__).parent.parent / "cache" / "tools.db"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Persistent SQLite-backed registry of user-defined tools."""

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self._conn = self._init_db(db_path)

    def register(
        self,
        name: str,
        description: str,
        parameters_schema: dict,
        webhook_url: str,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO tools
               (name, description, parameters_schema, webhook_url, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (name, description, json.dumps(parameters_schema), webhook_url,
             datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def list_tools(self) -> List[dict]:
        rows = self._conn.execute(
            "SELECT name, description, parameters_schema, webhook_url, created_at FROM tools"
        ).fetchall()
        return [
            {
                "name": r[0], "description": r[1],
                "parameters_schema": json.loads(r[2]),
                "webhook_url": r[3], "created_at": r[4],
            }
            for r in rows
        ]

    def delete(self, name: str) -> bool:
        cur = self._conn.execute("DELETE FROM tools WHERE name = ?", (name,))
        self._conn.commit()
        return cur.rowcount > 0

    def has_tools(self) -> bool:
        row = self._conn.execute("SELECT COUNT(*) FROM tools").fetchone()
        return (row[0] or 0) > 0

    @staticmethod
    def _init_db(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tools (
                name               TEXT PRIMARY KEY,
                description        TEXT NOT NULL,
                parameters_schema  TEXT NOT NULL,
                webhook_url        TEXT NOT NULL,
                created_at         TEXT NOT NULL
            )
        """)
        conn.commit()
        return conn


# ---------------------------------------------------------------------------
# Built-in tool implementations (no webhook needed)
# ---------------------------------------------------------------------------

_BUILTIN_TOOLS = {
    "get_current_datetime": {
        "description": "Returns the current UTC date and time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "calculate": {
        "description": "Evaluates a mathematical expression and returns the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A safe math expression, e.g. '2 + 2 * 10'",
                }
            },
            "required": ["expression"],
        },
    },
}


def _run_builtin(name: str, args: dict) -> Any:
    if name == "get_current_datetime":
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if name == "calculate":
        expr = args.get("expression", "")
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed_names.update({"abs": abs, "round": round, "int": int, "float": float})
        try:
            result = eval(expr, {"__builtins__": {}}, allowed_names)  # noqa: S307 — sandboxed
            return result
        except Exception as e:
            return f"Error: {e}"
    return "Unknown built-in tool"


def _call_webhook(webhook_url: str, tool_name: str, arguments: dict) -> Any:
    """POST to webhook URL and return parsed result."""
    import httpx  # type: ignore
    payload = {"tool_name": tool_name, "arguments": arguments}
    try:
        response = httpx.post(webhook_url, json=payload, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        return data.get("result", data)
    except Exception as exc:
        return f"Webhook error: {exc}"


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ToolUseStrategy(PromptStrategy):
    """
    Function-calling strategy. Uses the LLM's native tool/function API to
    let it decide which tool to invoke, then feeds results back for a
    final answer.

    Falls back to AdaptivePromptStrategy behavior when:
    - No tools are registered
    - Provider doesn't support function calling (mock)
    """

    def __init__(self, llm_client, evaluator=None, registry: Optional[ToolRegistry] = None):
        super().__init__(llm_client, evaluator)
        self.registry = registry or ToolRegistry()

    @property
    def name(self) -> str:
        return "tool_use"

    def build_prompt(self, query: str) -> str:
        return query

    def execute(
        self,
        query: str,
        model: "str | None" = None,
        baseline_model: "str | None" = None,
    ):
        if self.llm_client.provider == "mock":
            # Mock provider — skip tool use, return simple answer
            resp = self.llm_client.complete(query, model=model, baseline_model=baseline_model)
            conf = self.evaluator.score(query, resp.text)
            return resp.text, conf

        all_tools = self._get_all_tools()
        if not all_tools:
            # No tools — behave like adaptive strategy
            resp = self.llm_client.complete(query, model=model, baseline_model=baseline_model)
            conf = self.evaluator.score(query, resp.text)
            return resp.text, conf

        if self.llm_client.provider in ("openai", "groq"):
            return self._execute_openai_compat(query, model, baseline_model, all_tools)
        elif self.llm_client.provider == "gemini":
            return self._execute_gemini(query, model, baseline_model, all_tools)
        else:
            resp = self.llm_client.complete(query, model=model, baseline_model=baseline_model)
            conf = self.evaluator.score(query, resp.text)
            return resp.text, conf

    # ------------------------------------------------------------------
    # OpenAI / Groq function calling
    # ------------------------------------------------------------------

    def _execute_openai_compat(self, query, model, baseline_model, all_tools):
        tools_spec = [
            {"type": "function", "function": {"name": t["name"],
             "description": t["description"], "parameters": t["parameters"]}}
            for t in all_tools
        ]
        messages = [{"role": "user", "content": query}]
        effective_model = model or self.llm_client.model

        response = self.llm_client._client.chat.completions.create(
            model=effective_model,
            messages=messages,
            tools=tools_spec,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            # Execute tool calls
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self._execute_tool(tc.function.name, args, all_tools)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                })
            # Second LLM call with tool results
            final = self.llm_client._client.chat.completions.create(
                model=effective_model,
                messages=messages,
            )
            text = final.choices[0].message.content or ""
        else:
            text = msg.content or ""

        conf = self.evaluator.score(query, text)
        return text, conf

    # ------------------------------------------------------------------
    # Gemini function calling
    # ------------------------------------------------------------------

    def _execute_gemini(self, query, model, baseline_model, all_tools):
        from google import genai  # type: ignore
        from google.genai import types as gtypes  # type: ignore

        fn_declarations = []
        for t in all_tools:
            fn_declarations.append(
                gtypes.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=t["parameters"],
                )
            )
        tools = [gtypes.Tool(function_declarations=fn_declarations)]
        effective_model = model or self.llm_client.model

        response = self.llm_client._client.models.generate_content(
            model=effective_model,
            contents=query,
            config=gtypes.GenerateContentConfig(tools=tools, temperature=0.7),
        )

        # Check for function call in response
        fn_call = None
        for part in (response.candidates[0].content.parts if response.candidates else []):
            if hasattr(part, "function_call") and part.function_call:
                fn_call = part.function_call
                break

        if fn_call:
            args = dict(fn_call.args)
            result = self._execute_tool(fn_call.name, args, all_tools)
            fn_response = gtypes.Part(function_response=gtypes.FunctionResponse(
                name=fn_call.name,
                response={"result": result},
            ))
            contents = [
                query,
                response.candidates[0].content,
                gtypes.Content(parts=[fn_response], role="user"),
            ]
            final = self.llm_client._client.models.generate_content(
                model=effective_model,
                contents=contents,
                config=gtypes.GenerateContentConfig(temperature=0.7),
            )
            text = final.text or ""
        else:
            text = response.text or ""

        conf = self.evaluator.score(query, text)
        return text, conf

    # ------------------------------------------------------------------
    # Tool execution dispatch
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, args: dict, all_tools: list) -> Any:
        if name in _BUILTIN_TOOLS:
            return _run_builtin(name, args)
        # Look up webhook URL in registered tools
        for t in all_tools:
            if t["name"] == name and t.get("webhook_url"):
                return _call_webhook(t["webhook_url"], name, args)
        return f"Tool '{name}' not found."

    def _get_all_tools(self) -> List[Dict]:
        """Merge built-in tools with user-registered tools."""
        tools = [
            {"name": name, "description": info["description"],
             "parameters": info["parameters"], "webhook_url": None}
            for name, info in _BUILTIN_TOOLS.items()
        ]
        tools.extend(self.registry.list_tools())
        return tools
