#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from httpx import ASGITransport, AsyncClient
import pytest

from app.main import app


@pytest.mark.asyncio
async def test_predict_ready_or_info_ok():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # 使用ready或info
        resp = await ac.get("/api/v1/predict/ready")
        if resp.status_code == 404:
            resp = await ac.get("/api/v1/predict/info")
        assert resp.status_code in [200, 503]


@pytest.mark.asyncio
async def test_predict_qps_400_on_extra_field():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        payload = {
            "current_qps": 1.0,
            "prediction_hours": 1,
            "prediction_type_invalid": True,
        }
        resp = await ac.post("/api/v1/predict/qps", json=payload)
        assert resp.status_code == 400
        body = resp.json()
        assert body.get("code") == 400
