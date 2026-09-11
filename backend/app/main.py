"""Temporary Phase 1 application entry point."""

from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(title="Real-Time Face Recognition System", version="0.1.0")
app.include_router(router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return a small readiness response used by the phase smoke test."""
    return {"status": "ok", "phase": "1"}
