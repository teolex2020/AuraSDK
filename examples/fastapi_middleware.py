"""Aura + FastAPI: per-user memory middleware for AI chat backends.

Demonstrates:
  - Per-user namespace isolation (from auth header)
  - Auto-store of conversation turns
  - Context recall endpoint for prompt injection
  - Maintenance trigger

Requirements:
    pip install aura-memory fastapi uvicorn

Run:
    uvicorn examples.fastapi_middleware:app --reload
    # or: python examples/fastapi_middleware.py

Endpoints:
    POST /chat          — send message, get AI reply with memory context
    GET  /context       — recall memories for a query
    GET  /memories      — list stored memories
    POST /maintenance   — trigger maintenance cycle
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aura import Aura, Level

BRAIN_PATH = os.environ.get("AURA_BRAIN_PATH", "./fastapi_brain")

# ── Shared Aura instance ──

_brain: Optional[Aura] = None


def get_brain() -> Aura:
    """FastAPI dependency: returns the shared Aura instance."""
    assert _brain is not None, "Brain not initialized"
    return _brain


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open brain on startup, close on shutdown."""
    global _brain
    _brain = Aura(BRAIN_PATH)
    yield
    if _brain:
        _brain.run_maintenance()
        _brain.close()


app = FastAPI(
    title="Aura Memory API",
    description="Per-user persistent memory for AI agents",
    lifespan=lifespan,
)


# ── Auth: extract user_id from header ──

def get_user_id(x_user_id: str = Header(default="anonymous")) -> str:
    """Extract user ID from request header for namespace isolation.

    In production, replace this with JWT validation or session lookup.
    """
    if not x_user_id.strip():
        raise HTTPException(status_code=400, detail="X-User-Id header required")
    return x_user_id.strip()


# ── Request/Response models ──

class ChatRequest(BaseModel):
    message: str
    tags: list[str] = []


class ChatResponse(BaseModel):
    reply: str
    context_used: str
    memories_count: int


class ContextResponse(BaseModel):
    query: str
    context: str
    results: list[dict]


# ── Endpoints ──

@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    brain: Aura = Depends(get_brain),
    user_id: str = Depends(get_user_id),
):
    """Send a message and get a response with memory context.

    In production, replace the echo reply with your LLM call.
    The recalled context should be injected into the LLM system prompt.
    """
    # 1. Recall relevant context for this user
    context = brain.recall(
        req.message, token_budget=2000, namespace=user_id
    )

    # 2. Store the user message
    brain.store(
        f"User: {req.message}",
        level=Level.Working,
        tags=["conversation"] + req.tags,
        namespace=user_id,
    )

    # 3. Generate reply (replace with your LLM call)
    # Example with OpenAI:
    #   response = openai.chat.completions.create(
    #       model="gpt-4o-mini",
    #       messages=[
    #           {"role": "system", "content": f"You have memory:\n{context}"},
    #           {"role": "user", "content": req.message},
    #       ],
    #   )
    #   reply = response.choices[0].message.content
    reply = f"[Echo] {req.message} (context: {len(context)} chars recalled)"

    # 4. Store the reply
    brain.store(
        f"Assistant: {reply[:200]}",
        level=Level.Working,
        tags=["conversation"],
        namespace=user_id,
    )

    return ChatResponse(
        reply=reply,
        context_used=context[:500],
        memories_count=brain.count(),
    )


@app.get("/context", response_model=ContextResponse)
def get_context(
    query: str,
    top_k: int = 10,
    brain: Aura = Depends(get_brain),
    user_id: str = Depends(get_user_id),
):
    """Recall memories for a given query. Use this to build LLM prompts."""
    results = brain.recall_structured(query, top_k=top_k, namespace=user_id)
    context = brain.recall(query, token_budget=2000, namespace=user_id)
    return ContextResponse(query=query, context=context, results=results)


@app.get("/memories")
def list_memories(
    tags: Optional[str] = None,
    limit: int = 20,
    brain: Aura = Depends(get_brain),
    user_id: str = Depends(get_user_id),
):
    """List stored memories for the current user."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    results = brain.search(tags=tag_list, limit=limit, namespace=user_id)
    return [
        {
            "id": r.id,
            "content": r.content,
            "level": str(r.level),
            "strength": r.strength,
            "tags": r.tags,
        }
        for r in results
    ]


@app.post("/maintenance")
def run_maintenance(
    brain: Aura = Depends(get_brain),
    user_id: str = Depends(get_user_id),
):
    """Trigger a maintenance cycle (decay, consolidate, reflect)."""
    report = brain.run_maintenance()
    return {"status": "ok", "report": str(report)}


# ── Run directly ──

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("examples.fastapi_middleware:app", host="0.0.0.0", port=8080, reload=True)
