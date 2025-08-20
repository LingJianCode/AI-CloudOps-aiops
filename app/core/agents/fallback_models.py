#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 备用实现和数据模型 - 提供降级服务和数据结构定义
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger("aiops.fallback_models")


@dataclass
class SessionData:
    """会话数据模型"""
    session_id: str
    created_at: datetime
    last_activity: datetime
    history: List[str]


# ==================== 备用实现类 ====================

class FallbackEmbeddings(Embeddings):
    """备用嵌入实现 - 当主要嵌入服务不可用时使用"""

    def __init__(self):
        self.dimension = 384  # 默认维度

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入向量"""
        logger.warning("使用备用嵌入实现")
        embeddings = []
        for text in texts:
            # 简单的基于哈希的伪嵌入
            embedding = self._generate_pseudo_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """为查询生成嵌入向量"""
        logger.warning("使用备用查询嵌入实现")
        return self._generate_pseudo_embedding(text)

    def _generate_pseudo_embedding(self, text: str) -> List[float]:
        """生成伪嵌入向量"""
        import hashlib
        import struct

        # 使用文本哈希生成确定性的向量
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()

        # 将哈希转换为浮点数向量
        embedding = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                float_val = struct.unpack('f', chunk)[0]
                embedding.append(float_val)

        # 调整到目标维度
        while len(embedding) < self.dimension:
            embedding.extend(embedding[:self.dimension - len(embedding)])

        return embedding[:self.dimension]


class FallbackChatModel(BaseChatModel):
    """备用聊天模型 - 当主要LLM服务不可用时使用"""

    def __init__(self):
        super().__init__()
        self.model_name = "fallback-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ChatResult:
        """生成聊天响应"""
        logger.warning("使用备用聊天模型")

        # 分析最后一条用户消息
        user_message = ""
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                user_message = msg.content
                break

        # 生成简单的规则基础回答
        response_content = self._generate_rule_based_response(user_message)

        generation = ChatGeneration(
            message=type(messages[-1])(content=response_content),
            generation_info={"model": self.model_name}
        )

        return ChatResult(generations=[generation])

    def _generate_rule_based_response(self, user_input: str) -> str:
        """基于规则生成响应"""
        user_input_lower = user_input.lower()

        # 部署相关问题
        if any(keyword in user_input_lower for keyword in ["部署", "安装", "配置"]):
            return """**部署建议：**

1. **环境准备**
   - 确保系统资源充足
   - 检查网络连接和权限
   - 准备必要的配置文件

2. **部署步骤**
   - 下载或构建应用镜像
   - 配置环境变量和参数
   - 执行部署命令
   - 验证部署状态

3. **注意事项**
   - 遵循最佳实践
   - 做好备份和回滚准备
   - 监控部署过程

> 这是一个通用回答，具体步骤请参考相关技术文档。"""

        # 监控相关问题
        elif any(keyword in user_input_lower for keyword in ["监控", "告警", "指标"]):
            return """**监控建议：**

1. **基础监控**
   - CPU、内存、磁盘使用率
   - 网络流量和连接数
   - 应用响应时间

2. **告警设置**
   - 设定合理的阈值
   - 配置多级告警
   - 建立通知机制

3. **数据分析**
   - 定期检查监控数据
   - 分析趋势和异常
   - 优化监控策略

> 建议使用专业的监控工具如Prometheus + Grafana。"""

        # 故障排除相关问题
        elif any(keyword in user_input_lower for keyword in ["故障", "问题", "错误", "异常"]):
            return """**故障排除指南：**

1. **问题定位**
   - 收集错误信息和日志
   - 确定问题发生时间和范围
   - 检查相关组件状态

2. **诊断步骤**
   - 检查系统资源使用情况
   - 验证配置和权限
   - 测试网络连接
   - 查看应用日志

3. **解决方案**
   - 重启相关服务
   - 修复配置问题
   - 更新或回滚版本
   - 扩展系统资源

> 建议建立完善的故障处理流程和文档。"""

        # Kubernetes相关问题
        elif any(keyword in user_input_lower for keyword in ["kubernetes", "k8s", "pod", "deployment"]):
            return """**Kubernetes 运维建议：**

1. **基本操作**
   ```bash
   kubectl get pods
   kubectl describe pod <pod-name>
   kubectl logs <pod-name>
   ```

2. **常见问题**
   - Pod状态异常：检查资源配额和节点状态
   - 镜像拉取失败：验证镜像名称和网络
   - 配置错误：检查YAML文件格式

3. **最佳实践**
   - 使用资源限制和请求
   - 配置健康检查
   - 实施滚动更新策略

> 详细操作请参考Kubernetes官方文档。"""

        # 性能优化相关问题
        elif any(keyword in user_input_lower for keyword in ["性能", "优化", "慢", "卡顿"]):
            return """**性能优化建议：**

1. **系统层面**
   - 监控CPU、内存、磁盘I/O
   - 优化网络配置
   - 调整系统参数

2. **应用层面**
   - 代码优化和重构
   - 数据库查询优化
   - 缓存策略实施

3. **架构层面**
   - 负载均衡配置
   - 微服务拆分
   - 容器资源调优

> 建议进行性能测试和监控分析。"""

        # 默认回答
        else:
            return f"""感谢您的问题："{user_input}"

很抱歉，当前AI服务暂时不可用，我只能提供基础的技术建议：

**一般性运维建议：**
1. **监控为先** - 建立完善的监控体系
2. **自动化优先** - 减少手动操作，提高效率
3. **文档齐全** - 维护详细的操作文档
4. **备份策略** - 定期备份重要数据和配置
5. **安全第一** - 实施安全最佳实践

**获取更多帮助：**
- 查阅官方技术文档
- 联系技术支持团队
- 参考社区最佳实践

> 这是一个临时回答，建议您稍后重试或联系技术支持获得更准确的帮助。"""

    @property
    def _llm_type(self) -> str:
        return "fallback-chat"


# ==================== 工具函数 ====================

def _generate_fallback_answer() -> str:
    """生成降级答案"""
    return """很抱歉，AI助手服务当前不可用。

**可能的原因：**
- 服务正在维护中
- 网络连接问题
- 系统负载过高

**建议操作：**
1. 稍后重试
2. 检查网络连接
3. 联系技术支持

**临时解决方案：**
- 查阅技术文档
- 参考历史解决案例
- 咨询团队成员

感谢您的理解！"""


def _build_context_with_history(session: Optional[SessionData]) -> Optional[str]:
    """构建包含历史的上下文"""
    if not session or not session.history:
        return None

    # 获取最近的对话历史
    recent_history = session.history[-3:]  # 最近3轮对话

    if not recent_history:
        return None

    context_parts = ["## 对话历史"]
    for i, item in enumerate(recent_history, 1):
        context_parts.append(f"{i}. {item}")

    return "\n".join(context_parts)


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """清理和验证输入文本"""
    if not text:
        return ""

    # 移除潜在的有害字符
    import re
    cleaned = re.sub(r'[<>{}]', '', text)

    # 限制长度
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."

    return cleaned.strip()


def validate_session_id(session_id: str) -> bool:
    """验证会话ID格式 - 更宽松的验证"""
    if not session_id:
        return False

    # 长度检查
    if len(session_id) < 3 or len(session_id) > 128:
        return False

    try:
        # 尝试解析为UUID
        uuid.UUID(session_id)
        return True
    except ValueError:
        # 如果不是UUID格式，检查是否为合理的字符串
        # 允许字母、数字、连字符、下划线
        import re
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, session_id))


def create_session_id() -> str:
    """创建新的会话ID"""
    return str(uuid.uuid4())


def format_error_response(error_message: str, error_code: str = "UNKNOWN") -> Dict[str, Any]:
    """格式化错误响应"""
    return {
        "error": True,
        "error_code": error_code,
        "error_message": error_message,
        "timestamp": datetime.now().isoformat(),
        "suggestion": "请检查输入参数或稍后重试"
    }
