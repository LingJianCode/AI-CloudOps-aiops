#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from app.services.prediction_service import PredictionService


@pytest.mark.asyncio
async def test_predict_service_initialization():
    service = PredictionService()
    await service.initialize()
    assert service.is_initialized() is True


@pytest.mark.asyncio
async def test_predict_qps_basic_shape():
    service = PredictionService()
    await service.initialize()
    result = await service.predict_qps(current_qps=100.0, prediction_hours=6)
    assert isinstance(result, dict)
    assert "prediction_type" in result
    assert "predicted_data" in result
