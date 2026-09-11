"""FastAPI application entry point."""

import os

# OpenCV/FAISS and CPU PyTorch can load different OpenMP runtimes on Windows.
# This must be set before those libraries are imported by the API modules.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


app = FastAPI(title="Real-Time Face Recognition System", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return a small readiness response used by the phase smoke test."""
    return {"status": "ok", "phase": "1"}
