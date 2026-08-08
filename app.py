"""Local FrameGuard web server entry point."""

from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "frameguard.web:app",
        host=os.environ.get("FRAMEGUARD_HOST", "127.0.0.1"),
        port=int(os.environ.get("FRAMEGUARD_PORT", "7860")),
        reload=False,
    )
