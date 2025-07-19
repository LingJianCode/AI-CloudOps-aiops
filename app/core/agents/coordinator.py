#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
K8s多Agent协调器 - 负责协调所有agent的工作流
Author: AI Assistant
License: Apache 2.0
Description: 协调检测、策略、执行和验证的完整工作流
"""

import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, replace
from app.core.agents.detector import K8sDetectorAgent
from app.core.agents.strategist import K8sStrategistAgent
from app.core.agents.executor import K8sExecutorAgent
from app.services.notification import NotificationService

logger = logging.getLogger("aiops.coordinator")

@dataclass
class AgentState:
    """Agent状态数据结构"""
    deployment: str
    namespace: str
    issues: Dict[str, Any]
    strategy: Dict[str, Any]
    execution_result: Dict[str, Any]
    final_verification: Dict[str, Any]
    timestamp: str
    success: bool = False
    error_message: str = ""

class K8sCoordinatorAgent:
    """K8s多Agent协调器"""
    
    def __init__(self):
        self.detector = K8sDetectorAgent()
        self.strategist = K8sStrategistAgent()
        self.executor = K8sExecutorAgent()
        self.notification_service = NotificationService()
        self.workflow_history = []
    
    async def run_full_workflow(self, deployment: str, namespace: str = "default") -> Dict[str, Any]:
        """运行完整的修复工作流"""
        try:
            logger.info(f"🚀 开始K8s多Agent修复工作流: {deployment}/{namespace}")
            
            workflow_id = f"workflow_{int(asyncio.get_event_loop().time())}"
            start_time = datetime.now()
            
            # 步骤1: 检测问题
            logger.info("🔍 步骤1: 检测问题...")
            issues = await self.detector.detect_deployment_issues(deployment, namespace)
            
            if 'error' in issues:
                return {
                    'workflow_id': workflow_id,
                    'success': False,
                    'error': issues['error'],
                    'stage': 'detection'
                }
            
            if not issues['issues']:
                return {
                    'workflow_id': workflow_id,
                    'success': True,
                    'message': '未发现问题，无需修复',
                    'stage': 'detection'
                }
            
            # 步骤2: 制定策略
            logger.info("📋 步骤2: 制定修复策略...")
            all_issues = await self.detector.detect_all_issues(namespace)
            strategy = await self.strategist.analyze_issues(all_issues)
            
            if 'error' in strategy:
                return {
                    'workflow_id': workflow_id,
                    'success': False,
                    'error': strategy['error'],
                    'stage': 'strategy'
                }
            
            # 步骤3: 执行策略
            logger.info("⚙️ 步骤3: 执行修复策略...")
            execution_results = []
            
            for strategy_item in strategy.get('strategies', []):
                if strategy_item['target']['name'] == deployment:
                    execution_result = await self.executor.execute_strategy(strategy_item)
                    execution_results.append(execution_result)
            
            # 步骤4: 验证结果
            logger.info("✅ 步骤4: 验证修复结果...")
            final_issues = await self.detector.detect_deployment_issues(deployment, namespace)
            
            # 步骤5: 生成最终报告
            final_report = await self._generate_final_report(
                workflow_id, issues, strategy, execution_results, final_issues
            )
            
            # 保存工作流历史
            self.workflow_history.append({
                'workflow_id': workflow_id,
                'deployment': deployment,
                'namespace': namespace,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'issues_detected': len(issues['issues']),
                'strategies_created': len(strategy.get('strategies', [])),
                'executions': len(execution_results),
                'final_report': final_report
            })
            
            # 发送通知
            await self._send_workflow_notification(final_report)
            
            logger.info(f"✅ 工作流完成: {workflow_id}")
            return final_report
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            return {
                'workflow_id': f"workflow_{int(asyncio.get_event_loop().time())}",
                'success': False,
                'error': str(e),
                'stage': 'coordinator'
            }
    
    async def run_batch_workflow(self, namespace: str = "default") -> Dict[str, Any]:
        """批量处理命名空间内所有问题"""
        try:
            logger.info(f"🚀 开始批量修复工作流: {namespace}")
            
            workflow_id = f"batch_{int(asyncio.get_event_loop().time())}"
            start_time = datetime.now()
            
            # 检测所有问题
            all_issues = await self.detector.detect_all_issues(namespace)
            
            if 'error' in all_issues:
                return {'success': False, 'error': all_issues['error']}
            
            if all_issues['summary']['total_issues'] == 0:
                return {
                    'workflow_id': workflow_id,
                    'success': True,
                    'message': '未发现问题',
                    'issues_processed': 0
                }
            
            # 制定批量策略
            strategy = await self.strategist.analyze_issues(all_issues)
            
            # 执行所有策略
            execution_results = []
            for strategy_item in strategy.get('strategies', []):
                if strategy_item['auto_fix']:
                    result = await self.executor.execute_strategy(strategy_item)
                    execution_results.append(result)
            
            # 验证结果
            final_issues = await self.detector.detect_all_issues(namespace)
            
            batch_report = {
                'workflow_id': workflow_id,
                'namespace': namespace,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'initial_issues': all_issues['summary']['total_issues'],
                'fixable_issues': strategy['summary']['fixable_issues'],
                'executed_strategies': len(execution_results),
                'successful_executions': len([r for r in execution_results if r.get('success')]),
                'remaining_issues': final_issues['summary']['total_issues'],
                'success': final_issues['summary']['total_issues'] < all_issues['summary']['total_issues']
            }
            
            self.workflow_history.append(batch_report)
            
            return batch_report
            
        except Exception as e:
            logger.error(f"批量工作流失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _generate_final_report(self, workflow_id: str, initial_issues: Dict, 
                                   strategy: Dict, executions: List[Dict], 
                                   final_issues: Dict) -> Dict[str, Any]:
        """生成最终报告"""
        try:
            # 计算修复成功率
            total_issues = initial_issues['summary']['total_issues']
            remaining_issues = final_issues['summary']['total_issues']
            fixed_issues = max(0, total_issues - remaining_issues)
            success_rate = (fixed_issues / total_issues * 100) if total_issues > 0 else 0
            
            # 分析执行结果
            successful_executions = len([e for e in executions if e.get('success')])
            total_executions = len(executions)
            
            report = {
                'workflow_id': workflow_id,
                'timestamp': datetime.now().isoformat(),
                'success': remaining_issues < total_issues,
                'summary': {
                    'total_issues': total_issues,
                    'fixed_issues': fixed_issues,
                    'remaining_issues': remaining_issues,
                    'success_rate': round(success_rate, 2),
                    'executions': total_executions,
                    'successful_executions': successful_executions
                },
                'details': {
                    'initial_issues': initial_issues,
                    'strategy': strategy,
                    'execution_results': executions,
                    'final_issues': final_issues
                },
                'recommendations': await self._generate_recommendations(final_issues)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _generate_recommendations(self, final_issues: Dict) -> List[str]:
        """生成后续建议"""
        recommendations = []
        
        if 'error' in final_issues:
            recommendations.append("检查集群连接和权限")
            return recommendations
        
        if final_issues['summary']['total_issues'] > 0:
            remaining_types = set()
            for issue in final_issues.get('details', []):
                remaining_types.add(issue.get('sub_type', 'unknown'))
            
            for issue_type in remaining_types:
                if issue_type == 'image_pull_error':
                    recommendations.append("检查镜像仓库访问权限和镜像标签")
                elif issue_type == 'crash_loop':
                    recommendations.append("查看Pod日志，分析应用崩溃原因")
                elif issue_type == 'resource_pressure':
                    recommendations.append("增加节点资源或优化应用资源使用")
                elif issue_type == 'replica_mismatch':
                    recommendations.append("手动检查Deployment配置和节点状态")
        
        if not recommendations:
            recommendations.append("系统运行正常，建议定期监控")
        
        return recommendations
    
    async def _send_workflow_notification(self, report: Dict[str, Any]):
        """发送工作流完成通知"""
        try:
            summary = report['summary']
            message = f"""
🎯 K8s修复工作流完成

工作流ID: {report['workflow_id']}
修复成功率: {summary['success_rate']}%
修复问题数: {summary['fixed_issues']}/{summary['total_issues']}
执行策略数: {summary['successful_executions']}/{summary['executions']}

状态: {'✅ 成功' if report['success'] else '❌ 部分成功'}
"""
            
            await self.notification_service.send_notification(
                title="K8s修复工作流报告",
                message=message,
                notification_type="success" if report['success'] else "warning"
            )
            
        except Exception as e:
            logger.error(f"发送通知失败: {str(e)}")
    
    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """获取工作流历史"""
        return self.workflow_history
    
    async def health_check(self) -> Dict[str, Any]:
        """协调器健康检查"""
        try:
            # 检查各组件状态
            detector_healthy = hasattr(self.detector, 'k8s_service')
            strategist_healthy = hasattr(self.strategist, 'llm_service')
            executor_healthy = hasattr(self.executor, 'k8s_service')
            
            return {
                'healthy': all([detector_healthy, strategist_healthy, executor_healthy]),
                'components': {
                    'detector': detector_healthy,
                    'strategist': strategist_healthy,
                    'executor': executor_healthy
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def reset_workflow(self, deployment: str, namespace: str) -> Dict[str, Any]:
        """重置工作流（清理状态）"""
        try:
            # 这里可以添加清理逻辑
            return {
                'success': True,
                'message': f'工作流已重置: {deployment}/{namespace}',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}