"""Aura MCP Server — pure Python, zero extra dependencies.

Implements Model Context Protocol over stdio using JSON-RPC 2.0.
Works with Claude Desktop, Claude Code, and any MCP client.

Usage:
    python -m aura mcp [path]
"""

import json
import sys
from typing import Any

from aura import Aura, Level


def _parse_level(s: str) -> Level:
    return {
        "working": Level.Working,
        "decisions": Level.Decisions,
        "domain": Level.Domain,
        "identity": Level.Identity,
    }.get(s.lower(), Level.Working)


class AuraMcpServer:
    def __init__(self, path: str, password: str = None):
        if password:
            self.brain = Aura(path, password=password)
        else:
            self.brain = Aura(path)

    # ── MCP Tools ──

    def tool_recall(self, params: dict) -> str:
        query = params["query"]
        budget = params.get("token_budget", 2048)
        return self.brain.recall(query, token_budget=budget)

    def tool_recall_structured(self, params: dict) -> str:
        query = params["query"]
        top_k = params.get("top_k", 20)
        results = self.brain.recall_structured(query, top_k=top_k)
        items = []
        for r in results:
            items.append({
                "id": r["id"],
                "content": r["content"],
                "score": r["score"],
                "level": r.get("level", ""),
                "tags": r.get("tags", []),
            })
        return json.dumps(items)

    def tool_store(self, params: dict) -> str:
        content = params["content"]
        level = _parse_level(params["level"]) if "level" in params else None
        tags = params.get("tags")
        rid = self.brain.store(content, level=level, tags=tags)
        return json.dumps({"id": rid})

    def tool_store_code(self, params: dict) -> str:
        code = params["code"]
        language = params["language"]
        tags = params.get("tags", [])
        tags.extend(["code", language])
        if "filename" in params:
            tags.append(f"file:{params['filename']}")
        content = f"```{language}\n{code}\n```"
        rid = self.brain.store(content, level=Level.Domain, tags=tags)
        return json.dumps({"id": rid, "level": "DOMAIN"})

    def tool_store_decision(self, params: dict) -> str:
        content = f"DECISION: {params['decision']}"
        if params.get("reasoning"):
            content += f"\nREASONING: {params['reasoning']}"
        if params.get("alternatives"):
            content += f"\nALTERNATIVES: {', '.join(params['alternatives'])}"
        tags = params.get("tags", [])
        tags.append("decision")
        rid = self.brain.store(content, level=Level.Decisions, tags=tags)
        return json.dumps({"id": rid, "level": "DECISIONS"})

    def tool_search(self, params: dict) -> str:
        query = params.get("query")
        level = _parse_level(params["level"]) if "level" in params else None
        tags = params.get("tags")
        results = self.brain.search(query=query, level=level, tags=tags)
        items = [{"id": r.id, "content": r.content,
                  "level": r.level, "tags": r.tags} for r in results]
        return json.dumps(items)

    def tool_insights(self, params: dict) -> str:
        return json.dumps(self.brain.stats())

    def tool_consolidate(self, params: dict) -> str:
        result = self.brain.consolidate()
        return json.dumps({
            "merged": result.get("merged", 0),
            "checked": result.get("checked", 0),
        })

    # ── Tool definitions for MCP ──

    TOOLS = [
        {
            "name": "recall",
            "description": "Retrieve relevant memories as context for a query. Call BEFORE answering to check existing knowledge.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query to search memories."},
                    "token_budget": {"type": "integer", "description": "Maximum tokens in output (default: 2048)."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "recall_structured",
            "description": "Retrieve memories as structured data with scores. Use when you need individual records.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query."},
                    "top_k": {"type": "integer", "description": "Maximum results (default: 20)."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "store",
            "description": "Store a new memory. Levels: working (hours), decisions (days), domain (weeks), identity (months+).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The text content to store."},
                    "level": {"type": "string", "description": "Memory level: working, decisions, domain, or identity."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization."},
                },
                "required": ["content"],
            },
        },
        {
            "name": "store_code",
            "description": "Store a code snippet at DOMAIN level with language metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The source code."},
                    "language": {"type": "string", "description": "Programming language."},
                    "filename": {"type": "string", "description": "Optional filename."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags."},
                },
                "required": ["code", "language"],
            },
        },
        {
            "name": "store_decision",
            "description": "Store a decision with reasoning and rejected alternatives.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "description": "The decision made."},
                    "reasoning": {"type": "string", "description": "Reasoning behind it."},
                    "alternatives": {"type": "array", "items": {"type": "string"}, "description": "Alternatives considered."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags."},
                },
                "required": ["decision"],
            },
        },
        {
            "name": "search",
            "description": "Search memory by filters (exact/tag-based, not ranked). Use for browsing or counting.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text substring to match."},
                    "level": {"type": "string", "description": "Filter by level."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags."},
                },
            },
        },
        {
            "name": "insights",
            "description": "Get memory stats and health metrics.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "consolidate",
            "description": "Merge similar memory records (85%+ similarity) to reduce bloat.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]

    TOOL_MAP = {
        "recall": "tool_recall",
        "recall_structured": "tool_recall_structured",
        "store": "tool_store",
        "store_code": "tool_store_code",
        "store_decision": "tool_store_decision",
        "search": "tool_search",
        "insights": "tool_insights",
        "consolidate": "tool_consolidate",
    }

    # ── JSON-RPC / MCP Protocol ──

    def handle_request(self, msg: dict) -> dict | None:
        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            # Echo back the client's requested protocol version
            client_version = params.get("protocolVersion", "2024-11-05")
            return self._result(msg_id, {
                "protocolVersion": client_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "aura",
                    "version": "1.5.0",
                },
                "instructions": (
                    "Aura is a cognitive memory layer for AI agents. "
                    "Use 'recall' before answering to check existing context. "
                    "Use 'store' to remember facts, decisions, and patterns. "
                    "Levels: working (hours), decisions (days), domain (weeks), identity (months+)."
                ),
            })

        if method == "notifications/initialized":
            return None  # no response for notifications

        if method == "tools/list":
            return self._result(msg_id, {"tools": self.TOOLS})

        if method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            handler = self.TOOL_MAP.get(tool_name)
            if not handler:
                return self._error(msg_id, -32601, f"Unknown tool: {tool_name}")
            try:
                result_text = getattr(self, handler)(tool_args)
                return self._result(msg_id, {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": False,
                })
            except Exception as e:
                return self._result(msg_id, {
                    "content": [{"type": "text", "text": str(e)}],
                    "isError": True,
                })

        if method == "ping":
            return self._result(msg_id, {})

        # Unknown method
        if msg_id is not None:
            return self._error(msg_id, -32601, f"Method not found: {method}")
        return None

    def _result(self, msg_id: Any, result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _error(self, msg_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    def run_stdio(self):
        """Main loop: read JSON-RPC from stdin, write responses to stdout."""
        log = lambda msg: print(msg, file=sys.stderr, flush=True)
        log("Aura MCP server started (stdio)")

        while True:
            try:
                line = self._read_message()
                if line is None:
                    log("EOF on stdin, shutting down")
                    break

                log(f"<< {line[:200]}")
                msg = json.loads(line)
                response = self.handle_request(msg)
                if response is not None:
                    self._write_message(response)
                    log(f">> id={response.get('id')}")
            except json.JSONDecodeError as e:
                log(f"JSON parse error: {e}")
            except Exception as e:
                log(f"Error: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)

        self.brain.close()
        log("Aura MCP server stopped")

    def _read_line(self) -> str | None:
        """Read a single line from stdin, byte by byte to avoid buffering issues."""
        buf = bytearray()
        while True:
            b = sys.stdin.buffer.read(1)
            if not b:
                return None if not buf else buf.decode("utf-8")
            if b == b"\n":
                return buf.decode("utf-8").rstrip("\r")
            buf.extend(b)

    def _read_message(self) -> str | None:
        """Read a JSON-RPC message.

        Supports both Content-Length header framing (spec) and bare JSON lines.
        """
        first_line = self._read_line()
        if first_line is None:
            return None

        # If it starts with '{', it's a bare JSON message (no framing)
        if first_line.startswith("{"):
            return first_line

        # Otherwise, parse as Content-Length header framing
        headers = {}
        if ":" in first_line:
            key, value = first_line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        # Read remaining headers until empty line
        while True:
            line = self._read_line()
            if line is None:
                return None
            if line == "":
                break
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()

        length = int(headers.get("content-length", 0))
        if length == 0:
            return None

        body = sys.stdin.buffer.read(length)
        return body.decode("utf-8")

    def _write_message(self, msg: dict):
        """Write a JSON-RPC message as a single JSON line."""
        body = json.dumps(msg)
        sys.stdout.buffer.write(body.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()


def run_mcp(path: str = "./aura_brain", password: str = None):
    """Entry point for MCP server."""
    server = AuraMcpServer(path, password)
    server.run_stdio()
