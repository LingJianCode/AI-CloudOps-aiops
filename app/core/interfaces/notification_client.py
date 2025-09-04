#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Core层通知客户端接口定义与空实现
"""

from typing import Any, Dict, List, Protocol


class NotificationClient(Protocol):
    async def send_feishu_message(self, message: str, title: str = "AIOps通知", color: str = "blue") -> bool:
        ...


class NullNotificationClient:
    async def send_feishu_message(self, message: str, title: str = "AIOps通知", color: str = "blue") -> bool:
        return False


