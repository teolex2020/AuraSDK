"""Aura MCP HTTP Server — SSE transport for Make.com, n8n, and remote clients.

Implements MCP over HTTP+SSE as per the spec:
  GET  /sse          — SSE stream (client connects here)
  POST /message      — client sends JSON-RPC messages here

Also exposes plain REST for simple HTTP integrations:
  POST /store
  POST /recall
  GET  /search
  GET  /stats

Usage:
    python -m aura serve [path] [--host 0.0.0.0] [--port 8080]
"""

import asyncio
import json
import os
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from aura import Aura, Level

# ── App ──

app = FastAPI(title="Aura MCP Server", version="1.5.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Brain singleton ──

_brain: Aura | None = None

def get_brain() -> Aura:
    global _brain
    if _brain is None:
        path = os.environ.get("AURA_BRAIN_PATH", "./aura_brain")
        password = os.environ.get("AURA_PASSWORD")
        _brain = Aura(path, password=password) if password else Aura(path)
    return _brain


# ── SSE session store ──

_sessions: dict[str, asyncio.Queue] = {}


# ── MCP tool logic (shared with stdio server) ──

def _parse_level(s: str) -> Level:
    return {
        "working": Level.Working,
        "decisions": Level.Decisions,
        "domain": Level.Domain,
        "identity": Level.Identity,
    }.get(s.lower(), Level.Working)


def _handle_tool(name: str, args: dict) -> str:
    brain = get_brain()

    if name == "recall":
        return brain.recall(args["query"], token_budget=args.get("token_budget", 2048))

    if name == "recall_structured":
        results = brain.recall_structured(args["query"], top_k=args.get("top_k", 20))
        items = []
        for r in results:
            items.append({"id": r["id"], "content": r["content"],
                          "score": r["score"], "level": r.get("level", ""),
                          "tags": r.get("tags", [])})
        return json.dumps(items)

    if name == "store":
        level = _parse_level(args["level"]) if "level" in args else None
        rid = brain.store(args["content"], level=level, tags=args.get("tags"))
        return json.dumps({"id": rid})

    if name == "store_code":
        tags = args.get("tags", []) + ["code", args["language"]]
        if "filename" in args:
            tags.append(f"file:{args['filename']}")
        content = f"```{args['language']}\n{args['code']}\n```"
        rid = brain.store(content, level=Level.Domain, tags=tags)
        return json.dumps({"id": rid, "level": "DOMAIN"})

    if name == "store_decision":
        content = f"DECISION: {args['decision']}"
        if args.get("reasoning"):
            content += f"\nREASONING: {args['reasoning']}"
        if args.get("alternatives"):
            content += f"\nALTERNATIVES: {', '.join(args['alternatives'])}"
        tags = args.get("tags", []) + ["decision"]
        rid = brain.store(content, level=Level.Decisions, tags=tags)
        return json.dumps({"id": rid, "level": "DECISIONS"})

    if name == "search":
        level = _parse_level(args["level"]) if "level" in args else None
        results = brain.search(query=args.get("query"), level=level, tags=args.get("tags"))
        items = [{"id": r.id, "content": r.content,
                  "level": str(r.level), "tags": r.tags} for r in results]
        return json.dumps(items)

    if name == "insights":
        return json.dumps(brain.stats())

    if name == "consolidate":
        result = brain.consolidate()
        return json.dumps({"merged": result.get("merged", 0), "checked": result.get("checked", 0)})

    if name == "delete":
        deleted = brain.delete(args.get("id", ""))
        return json.dumps({"deleted": deleted, "id": args.get("id", "")})

    if name == "get":
        rec = brain.get(args.get("id", ""))
        if not rec:
            return json.dumps({"found": False})
        return json.dumps({
            "found": True,
            "id": rec.id,
            "content": rec.content,
            "level": str(rec.level),
            "tags": rec.tags,
            "strength": rec.strength,
            "source_type": rec.source_type,
        })

    if name == "maintain":
        report = brain.run_maintenance()
        return json.dumps({
            "total_records": report.total_records,
            "decayed": report.decay.decayed,
            "promoted": report.reflect.promoted,
            "archived": report.records_archived,
            "merged": report.consolidation.native_merged,
        })

    raise ValueError(f"Unknown tool: {name}")


TOOLS = [
    {"name": "recall", "description": "Retrieve relevant memories for a query.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "token_budget": {"type": "integer"}}, "required": ["query"]}},
    {"name": "recall_structured", "description": "Retrieve memories as structured data.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]}},
    {"name": "store", "description": "Store a new memory.", "inputSchema": {"type": "object", "properties": {"content": {"type": "string"}, "level": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["content"]}},
    {"name": "store_code", "description": "Store a code snippet at DOMAIN level.", "inputSchema": {"type": "object", "properties": {"code": {"type": "string"}, "language": {"type": "string"}, "filename": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["code", "language"]}},
    {"name": "store_decision", "description": "Store a decision with reasoning.", "inputSchema": {"type": "object", "properties": {"decision": {"type": "string"}, "reasoning": {"type": "string"}, "alternatives": {"type": "array", "items": {"type": "string"}}}, "required": ["decision"]}},
    {"name": "search", "description": "Search memory by filters.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "level": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "insights", "description": "Get memory health stats.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "consolidate", "description": "Merge similar memory records.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "delete", "description": "Delete a memory record by ID.", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "get", "description": "Get a memory record by ID with full metadata.", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "maintain", "description": "Run a full maintenance cycle (decay, promote, consolidate, archive).", "inputSchema": {"type": "object", "properties": {}}},
]


def _handle_jsonrpc(msg: dict) -> dict | None:
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params", {})

    def result(r): return {"jsonrpc": "2.0", "id": msg_id, "result": r}
    def error(code, text): return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": text}}

    if method == "initialize":
        return result({
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "aura", "version": "1.5.2"},
            "instructions": "Aura cognitive memory. Use 'recall' before answering. Use 'store' to remember facts.",
        })

    if method in ("notifications/initialized", "notifications/cancelled"):
        return None

    if method == "tools/list":
        return result({"tools": TOOLS})

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        try:
            text = _handle_tool(tool_name, tool_args)
            return result({"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as e:
            return result({"content": [{"type": "text", "text": str(e)}], "isError": True})

    if method == "ping":
        return result({})

    if msg_id is not None:
        return error(-32601, f"Method not found: {method}")
    return None


# ── SSE transport ──

@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE stream — MCP client connects here."""
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[session_id] = queue

    async def event_stream() -> AsyncGenerator[str, None]:
        # Send endpoint event so client knows where to POST
        yield f"event: endpoint\ndata: /message?session_id={session_id}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # keep-alive
        finally:
            _sessions.pop(session_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/message")
async def message_endpoint(request: Request, session_id: str):
    """Receive JSON-RPC from client, push response to SSE stream."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    body = await request.json()
    response = _handle_jsonrpc(body)
    if response is not None:
        await _sessions[session_id].put(response)

    return JSONResponse({"ok": True})


# ── Plain REST endpoints (for Make.com HTTP module / n8n) ──

class StoreRequest(BaseModel):
    content: str
    level: str | None = None
    tags: list[str] | None = None

class RecallRequest(BaseModel):
    query: str
    token_budget: int = 2048
    top_k: int = 20

@app.post("/store")
def rest_store(req: StoreRequest):
    brain = get_brain()
    level = _parse_level(req.level) if req.level else None
    rid = brain.store(req.content, level=level, tags=req.tags)
    return {"id": rid}

@app.post("/recall")
def rest_recall(req: RecallRequest):
    brain = get_brain()
    return {"context": brain.recall(req.query, token_budget=req.token_budget)}

@app.post("/recall_structured")
def rest_recall_structured(req: RecallRequest):
    brain = get_brain()
    results = brain.recall_structured(req.query, top_k=req.top_k)
    return {"results": results}

@app.get("/search")
def rest_search(query: str | None = None, level: str | None = None, tags: str | None = None):
    brain = get_brain()
    tag_list = tags.split(",") if tags else None
    lv = _parse_level(level) if level else None
    results = brain.search(query=query, level=lv, tags=tag_list)
    return {"results": [{"id": r.id, "content": r.content,
                         "level": str(r.level), "tags": r.tags} for r in results]}

@app.get("/stats")
def rest_stats():
    return get_brain().stats()

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.5.2"}


# ── Entry point ──

def run_http(path: str = "./aura_brain", host: str = "0.0.0.0", port: int = 8080, password: str | None = None):
    import uvicorn
    os.environ["AURA_BRAIN_PATH"] = path
    if password:
        os.environ["AURA_PASSWORD"] = password
    uvicorn.run(app, host=host, port=port)
