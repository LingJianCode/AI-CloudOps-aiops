#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_assistant_query_400_on_missing_question():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.post("/api/v1/assistant/query", json={})
        assert resp.status_code == 400
        body = resp.json()
        assert body.get("code") == 400
