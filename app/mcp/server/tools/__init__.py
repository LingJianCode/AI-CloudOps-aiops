"""
AI-CloudOps-aiops MCP工具集合
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: MCP服务器工具模块
"""

from .k8s_base_tool import K8sBaseTool
from .time_tool import TimeTool
from .system_info_tool import SystemInfoTool
from .file_tool import FileReadTool, FileListTool, FileStatsTool
from .calculator_tool import CalculatorTool, StatisticsTool, UnitConverterTool
from .k8s_cluster_check_tool import K8sClusterCheckTool
from .k8s_pod_tool import K8sPodTool
from .k8s_service_tool import K8sServiceTool
from .k8s_deployment_tool import K8sDeploymentTool
from .k8s_config_tool import K8sConfigTool
from .k8s_logs_tool import K8sLogsTool
from .k8s_monitor_tool import K8sMonitorTool
from .k8s_namespace_tool import K8sNamespaceTool
from .k8s_scaling_tool import K8sScalingTool
from .k8s_ingress_tool import K8sIngressTool
from .k8s_resource_tool import K8sResourceTool

__all__ = [
    'K8sBaseTool',
    'TimeTool',
    'SystemInfoTool',
    'FileReadTool',
    'FileListTool',
    'FileStatsTool',
    'CalculatorTool',
    'StatisticsTool',
    'UnitConverterTool',
    'K8sClusterCheckTool',
    'K8sPodTool',
    'K8sServiceTool',
    'K8sDeploymentTool',
    'K8sConfigTool',
    'K8sLogsTool',
    'K8sMonitorTool',
    'K8sNamespaceTool',
    'K8sScalingTool',
    'K8sIngressTool',
    'K8sResourceTool',
]

# 工具注册列表，用于MCP服务器自动加载
tools = [
    TimeTool(),
    SystemInfoTool(),
    FileReadTool(),
    FileListTool(),
    FileStatsTool(),
    CalculatorTool(),
    StatisticsTool(),
    UnitConverterTool(),
    K8sClusterCheckTool(),
    K8sPodTool(),
    K8sServiceTool(),
    K8sDeploymentTool(),
    K8sConfigTool(),
    K8sLogsTool(),
    K8sMonitorTool(),
    K8sNamespaceTool(),
    K8sScalingTool(),
    K8sIngressTool(),
    K8sResourceTool(),
]