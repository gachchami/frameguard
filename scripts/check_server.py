from __future__ import annotations

import os

import httpx

api_base = os.environ.get("FRAMEGUARD_API_BASE", "http://127.0.0.1:8091/v1")
health_url = f"{api_base.removesuffix('/v1')}/health"
response = httpx.get(health_url, timeout=10.0)
response.raise_for_status()
print(f"FrameGuard model server is healthy: {health_url}")
