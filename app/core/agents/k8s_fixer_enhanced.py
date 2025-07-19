#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
增强版Kubernetes智能修复代理
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 增强版的Kubernetes集群问题诊断和自动修复代理
"""

import logging
import time
import asyncio
from typing import Dict, Any, List
from app.services.kubernetes import KubernetesService
from app.services.llm import LLMService

logger = logging.getLogger("aiops.k8s_fixer_enhanced")

class EnhancedK8sFixerAgent:
    """增强版Kubernetes智能修复代理"""
    
    def __init__(self):
        self.k8s_service = KubernetesService()
        self.llm_service = LLMService()
        self.max_retries = 3
        self.retry_delay = 2
        
        # 定义问题识别规则
        self.problem_rules = {
            'crash_loop': {
                'patterns': ['CrashLoopBackOff', 'restart loop', 'continuous restart'],
                'severity': 'high',
                'auto_fix': True
            },
            'probe_failure': {
                'patterns': ['probe failed', 'health check failed', 'Unhealthy'],
                'severity': 'medium',
                'auto_fix': True
            },
            'resource_pressure': {
                'patterns': ['Insufficient memory', 'OutOfMemory', 'cpu throttling', 'resource pressure'],
                'severity': 'medium',
                'auto_fix': True
            },
            'image_pull_error': {
                'patterns': ['ImagePullBackOff', 'ErrImagePull', 'image not found'],
                'severity': 'high',
                'auto_fix': False
            }
        }
        
        # 修复策略模板
        self.fix_templates = {
            'nginx_probe_fix': {
                'livenessProbe': {
                    'httpGet': {'path': '/', 'port': 80},
                    'initialDelaySeconds': 10,
                    'periodSeconds': 10,
                    'failureThreshold': 3
                },
                'readinessProbe': {
                    'httpGet': {'path': '/', 'port': 80},
                    'initialDelaySeconds': 5,
                    'periodSeconds': 10,
                    'failureThreshold': 3
                }
            },
            'standard_resources': {
                'requests': {'memory': '64Mi', 'cpu': '50m'},
                'limits': {'memory': '128Mi', 'cpu': '100m'}
            }
        }
        
        logger.info("Enhanced K8s Fixer Agent initialized")
    
    async def analyze_and_fix_deployment(
        self, 
        deployment_name: str, 
        namespace: str, 
        error_description: str
    ) -> str:
        """智能分析并修复Kubernetes Deployment问题"""
        try:
            logger.info(f"🔍 开始智能分析Deployment: {deployment_name}/{namespace}")
            
            # 1. 收集完整上下文信息
            context = await self._gather_complete_context(deployment_name, namespace)
            if not context:
                return f"❌ 无法获取部署 {deployment_name} 的上下文信息"
            
            # 2. 智能问题识别
            problems = await self._intelligent_problem_detection(context, error_description)
            if not problems:
                return f"✅ 部署 {deployment_name} 未发现明显问题"
            
            # 3. 生成修复方案
            fix_plan = await self._generate_fix_plan(problems, context)
            if not fix_plan:
                return f"⚠️ 发现问题但无法自动生成修复方案: {', '.join([p['type'] for p in problems])}"
            
            # 4. 执行修复
            fix_result = await self._execute_fix_plan(deployment_name, namespace, fix_plan, context)
            
            # 5. 验证修复结果
            verification = await self._verify_fix_result(deployment_name, namespace, problems)
            
            return self._format_fix_report(fix_result, verification, problems)
            
        except Exception as e:
            logger.error(f"❌ 智能修复过程失败: {str(e)}")
            return f"修复失败: {str(e)}"
    
    async def _gather_complete_context(self, name: str, namespace: str) -> Dict[str, Any]:
        """收集完整的部署上下文信息"""
        try:
            deployment = await self.k8s_service.get_deployment(name, namespace)
            if not deployment:
                return {}
            
            pods = await self.k8s_service.get_pods(
                namespace=namespace,
                label_selector=f"app={name}"
            )
            
            events = await self.k8s_service.get_events(
                namespace=namespace,
                field_selector=f"involvedObject.name={name}",
                limit=50
            )
            
            # 获取部署状态详情
            status = await self.k8s_service.get_deployment_status(name, namespace)
            
            # 分析Pod状态
            pod_analysis = self._analyze_pod_status(pods)
            
            return {
                'deployment': deployment,
                'pods': pods,
                'events': events,
                'status': status,
                'pod_analysis': pod_analysis,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"收集上下文信息失败: {str(e)}")
            return {}
    
    def _analyze_pod_status(self, pods: List[Dict]) -> Dict[str, Any]:
        """分析Pod状态"""
        if not pods:
            return {'total': 0, 'ready': 0, 'issues': []}
        
        total_pods = len(pods)
        ready_pods = 0
        issues = []
        
        for pod in pods:
            status = pod.get('status', {})
            phase = status.get('phase', '')
            
            # 跳过非运行状态的Pod
            if phase not in ['Running', 'Pending']:
                continue
            
            # 检查就绪状态
            is_ready = False
            conditions = status.get('conditions', [])
            for condition in conditions:
                if condition.get('type') == 'Ready' and condition.get('status') == 'True':
                    is_ready = True
                    break
            
            if is_ready:
                ready_pods += 1
            
            # 检查容器状态
            container_statuses = status.get('container_statuses', [])
            for container_status in container_statuses:
                state = container_status.get('state', {})
                waiting = state.get('waiting', {})
                
                waiting_reason = waiting.get('reason', '')
                if waiting_reason:
                    issues.append({
                        'type': 'waiting',
                        'reason': waiting_reason,
                        'message': waiting.get('message', ''),
                        'pod': pod.get('metadata', {}).get('name', '')
                    })
                
                # 检查重启次数
                restart_count = container_status.get('restart_count', 0)
                if restart_count > 5:
                    issues.append({
                        'type': 'restart_loop',
                        'restart_count': restart_count,
                        'pod': pod.get('metadata', {}).get('name', '')
                    })
        
        return {
            'total': total_pods,
            'ready': ready_pods,
            'ready_ratio': ready_pods / total_pods if total_pods > 0 else 0,
            'issues': issues
        }
    
    async def _intelligent_problem_detection(self, context: Dict, error_desc: str) -> List[Dict[str, Any]]:
        """智能问题识别"""
        problems = []
        
        # 分析事件和错误描述
        text_to_analyze = error_desc.lower()
        events = context.get('events', [])
        
        # 从事件中提取问题
        for event in events:
            event_type = event.get('type', '').lower()
            reason = event.get('reason', '').lower()
            message = event.get('message', '').lower()
            
            text_to_analyze += f" {event_type} {reason} {message}"
        
        # 匹配问题规则
        for problem_type, rule in self.problem_rules.items():
            for pattern in rule['patterns']:
                if pattern.lower() in text_to_analyze:
                    problems.append({
                        'type': problem_type,
                        'pattern': pattern,
                        'severity': rule['severity'],
                        'auto_fix': rule['auto_fix'],
                        'context': self._extract_problem_context(problem_type, context)
                    })
                    break
        
        # 分析资源问题
        resource_issues = await self._analyze_resource_issues(context)
        problems.extend(resource_issues)
        
        # 分析探针问题
        probe_issues = await self._analyze_probe_issues(context)
        problems.extend(probe_issues)
        
        return problems
    
    async def _analyze_resource_issues(self, context: Dict) -> List[Dict[str, Any]]:
        """分析资源问题"""
        issues = []
        deployment = context.get('deployment', {})
        containers = deployment.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
        
        for container in containers:
            resources = container.get('resources', {})
            
            # 检查内存请求
            memory_request = resources.get('requests', {}).get('memory', '')
            if memory_request:
                memory_value = self._parse_memory_value(memory_request)
                if memory_value > 256:  # 超过256Mi
                    issues.append({
                        'type': 'high_memory_request',
                        'severity': 'medium',
                        'auto_fix': True,
                        'current': memory_request,
                        'suggested': '128Mi',
                        'container': container.get('name', 'main')
                    })
            
            # 检查CPU请求
            cpu_request = resources.get('requests', {}).get('cpu', '')
            if cpu_request:
                cpu_value = self._parse_cpu_value(cpu_request)
                if cpu_value > 300:  # 超过300m
                    issues.append({
                        'type': 'high_cpu_request',
                        'severity': 'medium',
                        'auto_fix': True,
                        'current': cpu_request,
                        'suggested': '100m',
                        'container': container.get('name', 'main')
                    })
        
        return issues
    
    async def _analyze_probe_issues(self, context: Dict) -> List[Dict[str, Any]]:
        """分析探针配置问题"""
        issues = []
        deployment = context.get('deployment', {})
        containers = deployment.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
        
        for container in containers:
            container_name = container.get('name', 'main')
            
            # 检查livenessProbe
            liveness = container.get('livenessProbe')
            if liveness:
                issues.extend(self._check_probe_config(liveness, 'livenessProbe', container_name))
            else:
                issues.append({
                    'type': 'missing_liveness_probe',
                    'severity': 'medium',
                    'auto_fix': True,
                    'container': container_name
                })
            
            # 检查readinessProbe
            readiness = container.get('readinessProbe')
            if readiness:
                issues.extend(self._check_probe_config(readiness, 'readinessProbe', container_name))
            else:
                issues.append({
                    'type': 'missing_readiness_probe',
                    'severity': 'medium',
                    'auto_fix': True,
                    'container': container_name
                })
        
        return issues
    
    def _check_probe_config(self, probe: Dict, probe_type: str, container: str) -> List[Dict[str, Any]]:
        """检查探针配置"""
        issues = []
        
        # 检查HTTP探针路径
        if 'httpGet' in probe:
            path = probe['httpGet'].get('path', '')
            if path in ['/nonexistent', '/healthz', '/health'] and self._is_nginx_container(container):
                issues.append({
                    'type': f'{probe_type}_path_error',
                    'severity': 'medium',
                    'auto_fix': True,
                    'current': path,
                    'suggested': '/',
                    'container': container
                })
        
        # 检查探针频率
        period = probe.get('periodSeconds', 10)
        if period < 5:
            issues.append({
                'type': f'{probe_type}_frequency_too_high',
                'severity': 'low',
                'auto_fix': True,
                'current': period,
                'suggested': 10,
                'container': container
            })
        
        return issues
    
    def _is_nginx_container(self, container_name: str) -> bool:
        """判断是否可能是nginx容器"""
        return 'nginx' in container_name.lower()
    
    def _parse_memory_value(self, memory_str: str) -> int:
        """解析内存值（转换为Mi）"""
        memory_str = memory_str.lower()
        if memory_str.endswith('mi'):
            return int(memory_str[:-2])
        elif memory_str.endswith('gi'):
            return int(memory_str[:-2]) * 1024
        elif memory_str.endswith('m'):
            return int(memory_str[:-1]) // 1000
        return 0
    
    def _parse_cpu_value(self, cpu_str: str) -> int:
        """解析CPU值（转换为m）"""
        cpu_str = cpu_str.lower()
        if cpu_str.endswith('m'):
            return int(cpu_str[:-1])
        elif cpu_str.isdigit():
            return int(cpu_str) * 1000
        return 0
    
    def _extract_problem_context(self, problem_type: str, context: Dict) -> Dict[str, Any]:
        """提取问题相关的上下文"""
        return {
            'deployment_name': context['deployment'].get('metadata', {}).get('name'),
            'namespace': context['deployment'].get('metadata', {}).get('namespace'),
            'pod_count': context['pod_analysis']['total'],
            'ready_count': context['pod_analysis']['ready']
        }
    
    async def _generate_fix_plan(self, problems: List[Dict], context: Dict) -> Dict[str, Any]:
        """生成修复方案"""
        if not problems:
            return {}
        
        fix_plan = {
            'deployment_name': context['deployment']['metadata']['name'],
            'namespace': context['deployment']['metadata']['namespace'],
            'fixes': [],
            'priority': 'medium'
        }
        
        # 按严重程度排序
        problems.sort(key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['severity']], reverse=True)
        
        # 为每个问题生成修复方案
        for problem in problems:
            if not problem['auto_fix']:
                continue
                
            fix = await self._create_fix_for_problem(problem, context)
            if fix:
                fix_plan['fixes'].append(fix)
        
        # 设置优先级
        if any(p['severity'] == 'high' for p in problems):
            fix_plan['priority'] = 'high'
        elif any(p['severity'] == 'medium' for p in problems):
            fix_plan['priority'] = 'medium'
        
        return fix_plan
    
    async def _create_fix_for_problem(self, problem: Dict, context: Dict) -> Dict[str, Any]:
        """为特定问题创建修复方案"""
        problem_type = problem['type']
        
        if 'probe' in problem_type:
            return await self._create_probe_fix(problem, context)
        elif 'resource' in problem_type:
            return await self._create_resource_fix(problem, context)
        elif problem_type == 'restart_loop':
            return await self._create_restart_fix(problem, context)
        
        return {}
    
    async def _create_probe_fix(self, problem: Dict, context: Dict) -> Dict[str, Any]:
        """创建探针修复方案"""
        container_name = problem.get('container', 'main')
        
        if 'missing' in problem['type']:
            probe_type = 'livenessProbe' if 'liveness' in problem['type'] else 'readinessProbe'
            return {
                'type': 'add_probe',
                'container': container_name,
                'probe_type': probe_type,
                'config': self.fix_templates['nginx_probe_fix'][probe_type]
            }
        elif 'path' in problem['type']:
            probe_type = 'livenessProbe' if 'liveness' in problem['type'] else 'readinessProbe'
            return {
                'type': 'update_probe_path',
                'container': container_name,
                'probe_type': probe_type,
                'path': problem['suggested']
            }
        
        return {}
    
    async def _create_resource_fix(self, problem: Dict, context: Dict) -> Dict[str, Any]:
        """创建资源修复方案"""
        return {
            'type': 'update_resources',
            'container': problem['container'],
            'resource_type': 'memory' if 'memory' in problem['type'] else 'cpu',
            'current': problem['current'],
            'suggested': problem['suggested']
        }
    
    async def _create_restart_fix(self, problem: Dict, context: Dict) -> Dict[str, Any]:
        """创建重启修复方案"""
        return {
            'type': 'restart_deployment',
            'reason': 'high restart count detected'
        }
    
    async def _execute_fix_plan(self, name: str, namespace: str, plan: Dict, context: Dict) -> Dict[str, Any]:
        """执行修复方案"""
        results = {
            'success': True,
            'actions': [],
            'errors': []
        }
        
        for fix in plan.get('fixes', []):
            try:
                result = await self._execute_single_fix(name, namespace, fix, context)
                if result['success']:
                    results['actions'].append(result['action'])
                else:
                    results['errors'].append(result['error'])
                    results['success'] = False
            except Exception as e:
                results['errors'].append(str(e))
                results['success'] = False
        
        return results
    
    async def _execute_single_fix(self, name: str, namespace: str, fix: Dict, context: Dict) -> Dict[str, Any]:
        """执行单个修复操作"""
        fix_type = fix['type']
        
        try:
            if fix_type == 'add_probe':
                return await self._add_probe(name, namespace, fix)
            elif fix_type == 'update_probe_path':
                return await self._update_probe_path(name, namespace, fix)
            elif fix_type == 'update_resources':
                return await self._update_resources(name, namespace, fix)
            elif fix_type == 'restart_deployment':
                return await self._restart_deployment(name, namespace, fix)
            else:
                return {'success': False, 'error': f"未知修复类型: {fix_type}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _add_probe(self, name: str, namespace: str, fix: Dict) -> Dict[str, Any]:
        """添加探针"""
        probe_type = fix['probe_type']
        container_name = fix['container']
        
        patch = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": container_name,
                            probe_type: fix['config']
                        }]
                    }
                }
            }
        }
        
        success = await self.k8s_service.patch_deployment(name, patch, namespace)
        if success:
            return {'success': True, 'action': f"添加{probe_type}到容器{container_name}"}
        else:
            return {'success': False, 'error': f"添加{probe_type}失败"}
    
    async def _update_probe_path(self, name: str, namespace: str, fix: Dict) -> Dict[str, Any]:
        """更新探针路径"""
        probe_type = fix['probe_type']
        container_name = fix['container']
        
        patch = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": container_name,
                            probe_type: {
                                "httpGet": {"path": fix['path'], "port": 80}
                            }
                        }]
                    }
                }
            }
        }
        
        success = await self.k8s_service.patch_deployment(name, patch, namespace)
        if success:
            return {'success': True, 'action': f"更新{probe_type}路径为{fix['path']}"}
        else:
            return {'success': False, 'error': f"更新{probe_type}路径失败"}
    
    async def _update_resources(self, name: str, namespace: str, fix: Dict) -> Dict[str, Any]:
        """更新资源配置"""
        container_name = fix['container']
        resource_type = fix['resource_type']
        
        resource_key = 'memory' if resource_type == 'memory' else 'cpu'
        
        patch = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": container_name,
                            "resources": {
                                "requests": {resource_key: fix['suggested']},
                                "limits": {resource_key: fix['suggested']}
                            }
                        }]
                    }
                }
            }
        }
        
        success = await self.k8s_service.patch_deployment(name, patch, namespace)
        if success:
            return {'success': True, 'action': f"更新{resource_type}资源为{fix['suggested']}"}
        else:
            return {'success': False, 'error': f"更新{resource_type}资源失败"}
    
    async def _restart_deployment(self, name: str, namespace: str, fix: Dict) -> Dict[str, Any]:
        """重启部署"""
        success = await self.k8s_service.restart_deployment(name, namespace)
        if success:
            return {'success': True, 'action': "重启部署"}
        else:
            return {'success': False, 'error': "重启部署失败"}
    
    async def _verify_fix_result(self, name: str, namespace: str, original_problems: List[Dict]) -> str:
        """验证修复结果"""
        try:
            # 等待修复生效
            await asyncio.sleep(5)
            
            # 重新收集上下文
            new_context = await self._gather_complete_context(name, namespace)
            if not new_context:
                return "无法验证修复结果"
            
            # 重新分析是否有相同问题
            new_problems = await self._intelligent_problem_detection(new_context, "")
            
            # 检查原始问题是否已解决
            resolved = []
            remaining = []
            
            for original in original_problems:
                original_type = original['type']
                found = False
                for new_problem in new_problems:
                    if new_problem['type'] == original_type:
                        remaining.append(original_type)
                        found = True
                        break
                if not found:
                    resolved.append(original_type)
            
            # 检查Pod状态
            pod_analysis = new_context['pod_analysis']
            ready_ratio = pod_analysis['ready_ratio']
            
            report = f"""
修复验证报告:
- 就绪状态: {pod_analysis['ready']}/{pod_analysis['total']} Pod已就绪 ({ready_ratio:.0%})
- 已解决问题: {', '.join(resolved) if resolved else '无'}
- 仍存在问题: {', '.join(remaining) if remaining else '无'}
            """
            
            if ready_ratio >= 0.9 and not remaining:
                return report + "\n✅ 修复成功！"
            elif ready_ratio >= 0.7:
                return report + "\n⚠️ 部分修复成功，建议手动检查"
            else:
                return report + "\n❌ 修复效果不佳，需要进一步处理"
                
        except Exception as e:
            return f"验证失败: {str(e)}"
    
    def _format_fix_report(self, fix_result: Dict, verification: str, problems: List[Dict]) -> str:
        """格式化修复报告"""
        report = f"""
🎯 智能修复报告
================
发现的问题:
"""
        
        for problem in problems:
            report += f"- {problem['type']}: {problem.get('description', '配置问题')}\n"
        
        if fix_result['actions']:
            report += f"\n✅ 执行的操作:\n"
            for action in fix_result['actions']:
                report += f"- {action}\n"
        
        if fix_result['errors']:
            report += f"\n❌ 执行失败:\n"
            for error in fix_result['errors']:
                report += f"- {error}\n"
        
        report += f"\n{verification}"
        
        return report
    
    async def diagnose_deployment_health(self, name: str, namespace: str) -> Dict[str, Any]:
        """诊断部署健康状态"""
        try:
            context = await self._gather_complete_context(name, namespace)
            if not context:
                return {'healthy': False, 'error': '无法获取部署信息'}
            
            problems = await self._intelligent_problem_detection(context, "")
            
            return {
                'healthy': len(problems) == 0,
                'problems': problems,
                'pod_status': context['pod_analysis'],
                'recommendations': await self._generate_recommendations(problems, context)
            }
            
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def _generate_recommendations(self, problems: List[Dict], context: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        for problem in problems:
            if problem['type'] == 'crash_loop':
                recommendations.append("检查容器日志和配置，可能存在应用错误或配置问题")
            elif 'probe' in problem['type']:
                recommendations.append("检查探针配置是否合理，确保应用启动时间充足")
            elif 'resource' in problem['type']:
                recommendations.append("调整资源配置，确保在节点资源范围内")
        
        return recommendations
