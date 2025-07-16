"""
AI-CloudOps-aiops MCP工具集合
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: MCP服务器工具模块
"""

from .time_tool import TimeTool
from .system_info_tool import SystemInfoTool
from .file_tool import FileReadTool, FileListTool, FileStatsTool
from .calculator_tool import CalculatorTool, StatisticsTool, UnitConverterTool

__all__ = [
    'TimeTool',
    'SystemInfoTool',
    'FileReadTool',
    'FileListTool',
    'FileStatsTool',
    'CalculatorTool',
    'StatisticsTool',
    'UnitConverterTool'
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
    UnitConverterTool()
]