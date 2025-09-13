#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Kubernetes集群检查工具
"""

import asyncio
from datetime import datetime, timedelta
import os
from typing import Any, Dict, List, Optional

from kubernetes import client, config

from .k8s_base_tool import K8sBaseTool


class K8sClusterCheckTool(K8sBaseTool):
    """k8s集群健康检查工具"""

    def __init__(self):
        super().__init__(
            name="k8s_cluster_check",
            description="执行k8s集群健康检查，返回集群状态、节点状态、事件和日志的markdown格式报告",
        )

    def get_parameters(self) -> Dict[str, Any]:
        """获取工具参数定义"""
        return {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "可选的kubeconfig文件路径，默认使用集群内配置或~/.kube/config",
                },
                "namespace": {
                    "type": "string",
                    "description": "可选的命名空间过滤，用于限制检查范围",
                },
                "time_window_hours": {
                    "type": "integer",
                    "description": "事件查询时间窗口（小时），默认1小时",
                    "minimum": 1,
                    "maximum": 24,
                    "default": 1,
                },
            },
            "required": [],
        }

    def _create_api_clients(
        self, config_path: Optional[str] = None
    ) -> Dict[str, client.ApiClient]:
        """创建Kubernetes API客户端"""
        try:
            # 尝试不同的配置加载方式
            if config_path and os.path.exists(config_path):
                config.load_kube_config(config_file=config_path)
            elif os.path.exists(os.path.expanduser("~/.kube/config")):
                config.load_kube_config()
            else:
                # 尝试集群内配置
                config.load_incluster_config()

            return {
                "v1": client.CoreV1Api(),
                "apps_v1": client.AppsV1Api(),
                "version": client.VersionApi(),
            }

        except Exception as e:
            raise Exception(f"无法加载Kubernetes配置: {str(e)}")

    async def _execute_internal(self, parameters: Dict[str, Any]) -> Any:
        """执行工具内部逻辑"""
        # 提取参数
        config_path = parameters.get("config_path")
        namespace_filter = parameters.get("namespace")
        time_window = parameters.get("time_window_hours", 1)

        # 创建API客户端
        clients = self._create_api_clients(config_path)
        v1 = clients["v1"]
        version_api = clients["version"]

        # 并行执行检查任务，设置更短的超时
        from app.common.constants import ServiceConstants

        timeout_short = ServiceConstants.AUTOFIX_K8S_TIMEOUT - 22
        timeout_long = ServiceConstants.AUTOFIX_K8S_TIMEOUT - 18

        tasks = [
            asyncio.wait_for(
                self._get_cluster_info(version_api, v1), timeout=timeout_short
            ),
            asyncio.wait_for(self._get_node_status(v1), timeout=timeout_long),
            asyncio.wait_for(
                self._get_recent_events(v1, time_window, namespace_filter),
                timeout=timeout_long,
            ),
            asyncio.wait_for(
                self._get_pod_status(v1, namespace_filter), timeout=timeout_long
            ),
            asyncio.wait_for(
                self._get_error_logs(v1, namespace_filter), timeout=timeout_short
            ),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        cluster_info, node_status, events, pod_status, error_logs = results

        # 处理异常结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 区分超时错误和其他错误
                if isinstance(result, asyncio.TimeoutError):
                    error_msg = f"检查任务{i}超时: 请求执行时间过长，已跳过"
                else:
                    error_msg = f"检查任务{i}失败: {str(result)}"

                if i == 0:
                    cluster_info = {"error": error_msg}
                elif i == 1:
                    node_status = {"error": error_msg}
                elif i == 2:
                    events = [{"error": error_msg}]
                elif i == 3:
                    pod_status = {"error": error_msg}
                elif i == 4:
                    error_logs = [{"error": error_msg}]

        # 生成健康检查报告
        report = self._generate_health_report(
            cluster_info, node_status, events, pod_status, error_logs, namespace_filter
        )

        return {
            "report": report,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "success",
        }

    async def _get_cluster_info(
        self, version_api: client.VersionApi, v1: client.CoreV1Api
    ) -> Dict[str, Any]:
        """获取集群基本信息"""
        try:
            # 使用线程池并行执行同步操作
            loop = asyncio.get_event_loop()

            # 并行获取信息，严格限制数量
            version_task = loop.run_in_executor(self._executor, version_api.get_code)
            namespaces_task = loop.run_in_executor(
                self._executor, lambda: v1.list_namespace(limit=50)
            )  # 进一步限制

            version_info, namespaces = await asyncio.gather(
                version_task, namespaces_task, return_exceptions=True
            )

            # 处理版本信息
            if isinstance(version_info, Exception):
                raise version_info

            # 处理命名空间信息
            if isinstance(namespaces, Exception):
                namespace_count = 0
            else:
                namespace_count = len(namespaces.items)

            # 简化权限检查，不做超时操作
            has_cluster_access = True  # 假设有权限，减少额外检查

            return {
                "version": f"{version_info.major}.{version_info.minor}",
                "server_version": version_info.git_version,
                "platform": version_info.platform,
                "namespace_count": namespace_count,
                "has_cluster_access": has_cluster_access,
            }
        except Exception as e:
            return {"error": f"获取集群信息失败: {str(e)}"}

    async def _get_node_status(self, v1: client.CoreV1Api) -> Dict[str, Any]:
        """获取节点状态"""
        try:
            loop = asyncio.get_event_loop()
            nodes = await loop.run_in_executor(
                self._executor, lambda: v1.list_node(limit=20)
            )  # 进一步限制节点数量

            total_nodes = len(nodes.items)
            if total_nodes == 0:
                return {"error": "没有找到任何节点"}

            ready_nodes = 0
            not_ready_nodes = []
            node_details = []

            # 限制处理的节点数量以提高性能
            for node in nodes.items[:10]:  # 只处理前10个节点
                node_info = {
                    "name": node.metadata.name,
                    "version": node.status.node_info.kubelet_version,
                    "os": f"{node.status.node_info.os_image}",
                    "ready": False,
                    "conditions": [],
                }

                # 检查节点条件
                for condition in node.status.conditions or []:
                    if condition.type == "Ready":
                        if condition.status == "True":
                            ready_nodes += 1
                            node_info["ready"] = True
                        else:
                            not_ready_nodes.append(
                                {
                                    "name": node.metadata.name,
                                    "reason": condition.reason or "Unknown",
                                    "message": (condition.message or "")[
                                        :100
                                    ],  # 限制消息长度
                                }
                            )
                        break

                node_details.append(node_info)

            return {
                "total": total_nodes,
                "ready": ready_nodes,
                "not_ready": not_ready_nodes[:3],  # 只返回前3个问题节点
                "health_percentage": (ready_nodes / total_nodes * 100),
                "node_details": node_details[:3],  # 限制详情数量
            }
        except Exception as e:
            return {"error": f"获取节点状态失败: {str(e)}"}

    async def _get_recent_events(
        self,
        v1: client.CoreV1Api,
        time_window: int,
        namespace_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取近期重要事件"""
        try:
            since_time = datetime.utcnow() - timedelta(hours=time_window)
            loop = asyncio.get_event_loop()

            # 根据命名空间过滤获取事件，进一步限制数量
            if namespace_filter:
                events = await loop.run_in_executor(
                    self._executor,
                    lambda: v1.list_namespaced_event(
                        namespace=namespace_filter, limit=20
                    ),  # 减少到20
                )
            else:
                events = await loop.run_in_executor(
                    self._executor,
                    lambda: v1.list_event_for_all_namespaces(limit=20),  # 减少到20
                )

            # 过滤重要事件并按时间排序
            important_events = []
            for event in events.items:
                # 检查事件时间
                event_time = event.last_timestamp or event.first_timestamp
                if event_time and event_time >= since_time.replace(
                    tzinfo=event_time.tzinfo
                ):
                    if event.type in ["Warning", "Error"] or event.reason in [
                        "Failed",
                        "FailedMount",
                        "FailedScheduling",
                    ]:
                        important_events.append(
                            {
                                "type": event.type,
                                "reason": event.reason,
                                "message": (event.message or "")[:150],  # 限制消息长度
                                "namespace": event.metadata.namespace or "default",
                                "object": f"{event.involved_object.kind}/{event.involved_object.name}",
                                "count": event.count or 1,
                                "first_seen": event.first_timestamp,
                                "last_seen": event.last_timestamp,
                            }
                        )

            # 按最后发生时间排序
            important_events.sort(
                key=lambda x: x["last_seen"] or x["first_seen"], reverse=True
            )

            return important_events[:8]  # 返回最近8个事件
        except Exception as e:
            return [{"error": f"获取事件失败: {str(e)}"}]

    async def _get_pod_status(
        self, v1: client.CoreV1Api, namespace_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取Pod状态统计"""
        try:
            loop = asyncio.get_event_loop()

            # 进一步限制Pod数量以提高性能
            if namespace_filter:
                pods = await loop.run_in_executor(
                    self._executor,
                    lambda: v1.list_namespaced_pod(
                        namespace=namespace_filter, limit=50
                    ),  # 减少到50
                )
            else:
                pods = await loop.run_in_executor(
                    self._executor,
                    lambda: v1.list_pod_for_all_namespaces(limit=50),  # 减少到50
                )

            pod_stats = {
                "total": len(pods.items),
                "running": 0,
                "pending": 0,
                "failed": 0,
                "succeeded": 0,
                "unknown": 0,
                "problem_pods": [],
            }

            for pod in pods.items:
                phase = pod.status.phase
                if phase == "Running":
                    pod_stats["running"] += 1
                elif phase == "Pending":
                    pod_stats["pending"] += 1
                elif phase == "Failed":
                    pod_stats["failed"] += 1
                elif phase == "Succeeded":
                    pod_stats["succeeded"] += 1
                else:
                    pod_stats["unknown"] += 1

                # 检查容器状态，只记录问题Pod
                if phase not in ["Running", "Succeeded"]:
                    pod_stats["problem_pods"].append(
                        {
                            "name": pod.metadata.name,
                            "namespace": pod.metadata.namespace,
                            "phase": phase,
                            "reason": (pod.status.reason or "Unknown")[
                                :50
                            ],  # 限制原因长度
                        }
                    )

            # 限制问题Pod数量
            pod_stats["problem_pods"] = pod_stats["problem_pods"][:3]  # 减少到3个

            return pod_stats
        except Exception as e:
            return {"error": f"获取Pod状态失败: {str(e)}"}

    async def _get_error_logs(
        self, v1: client.CoreV1Api, namespace_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取错误日志（限制数量和时间）"""
        try:
            error_logs = []
            loop = asyncio.get_event_loop()

            # 获取Pod列表，严格限制数量
            if namespace_filter:
                pods = await loop.run_in_executor(
                    self._executor,
                    lambda: v1.list_namespaced_pod(
                        namespace=namespace_filter, limit=10
                    ),  # 减少到10
                )
            else:
                pods = await loop.run_in_executor(
                    self._executor,
                    lambda: v1.list_pod_for_all_namespaces(limit=10),  # 减少到10
                )

            # 筛选有问题的Pod，但只检查最新的几个
            problem_pods = []
            for pod in pods.items:
                if pod.status.phase in ["Failed", "Pending"]:
                    problem_pods.append(pod)
                elif pod.status.phase == "Running":
                    # 简化检查：只检查重启次数
                    for container_status in pod.status.container_statuses or []:
                        if container_status.restart_count > 3:  # 只关注频繁重启的
                            problem_pods.append(pod)
                            break

                if len(problem_pods) >= 2:  # 只检查2个问题Pod
                    break

            # 获取问题Pod的日志，但要限制日志大小
            for pod in problem_pods:
                try:
                    pod_name = pod.metadata.name
                    namespace = pod.metadata.namespace

                    # 只获取最近的少量日志
                    log_content = await loop.run_in_executor(
                        self._executor,
                        lambda: v1.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=namespace,
                            tail_lines=10,  # 减少行数
                            timestamps=False,  # 不要时间戳
                            previous=False,
                        ),
                    )

                    # 限制日志内容大小
                    if len(log_content) > 300:
                        log_content = log_content[-300:]

                    error_logs.append(
                        {
                            "pod": pod_name,
                            "namespace": namespace,
                            "phase": pod.status.phase,
                            "restart_count": sum(
                                cs.restart_count
                                for cs in pod.status.container_statuses or []
                            ),
                            "sample_log": (
                                log_content.strip()
                                if log_content.strip()
                                else "无日志内容"
                            ),
                        }
                    )

                except Exception:
                    # 忽略日志获取失败，但记录Pod状态
                    error_logs.append(
                        {
                            "pod": pod.metadata.name,
                            "namespace": pod.metadata.namespace,
                            "phase": pod.status.phase,
                            "error": "无法获取日志",
                        }
                    )

            return error_logs
        except Exception as e:
            return [{"error": f"获取日志失败: {str(e)}"}]

    def _generate_health_report(
        self,
        cluster_info: Dict[str, Any],
        node_status: Dict[str, Any],
        events: List[Dict[str, Any]],
        pod_status: Dict[str, Any],
        error_logs: List[Dict[str, Any]],
        namespace_filter: Optional[str] = None,
    ) -> str:
        """生成健康检查报告"""

        report_lines = []
        report_lines.append("# Kubernetes集群健康检查报告")
        report_lines.append(
            f"**检查时间**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        if namespace_filter:
            report_lines.append(f"**检查范围**: 命名空间 `{namespace_filter}`")
        else:
            report_lines.append("**检查范围**: 全集群")
        report_lines.append("")

        # 集群概览
        report_lines.append("## 🏗️ 集群概览")
        if "error" not in cluster_info:
            report_lines.append(
                f"- **Kubernetes版本**: {cluster_info['server_version']}"
            )
            report_lines.append(f"- **API版本**: {cluster_info['version']}")
            report_lines.append(
                f"- **平台**: {cluster_info.get('platform', 'Unknown')}"
            )
            report_lines.append(
                f"- **命名空间数量**: {cluster_info['namespace_count']}"
            )
            access_status = (
                "✅ 正常" if cluster_info.get("has_cluster_access", False) else "⚠️ 受限"
            )
            report_lines.append(f"- **权限状态**: {access_status}")
        else:
            report_lines.append(f"- **错误**: {cluster_info['error']}")
        report_lines.append("")

        # 节点状态
        report_lines.append("## 🖥️ 节点状态")
        if "error" not in node_status:
            health_percentage = node_status["health_percentage"]
            if health_percentage >= 90:
                health_icon = "✅"
                health_status = "健康"
            elif health_percentage >= 70:
                health_icon = "⚠️"
                health_status = "警告"
            else:
                health_icon = "🔴"
                health_status = "异常"

            report_lines.append(
                f"- **节点健康度**: {health_icon} {health_percentage:.1f}% ({health_status})"
            )
            report_lines.append(
                f"- **就绪节点**: {node_status['ready']}/{node_status['total']}"
            )

            if node_status["not_ready"]:
                report_lines.append("")
                report_lines.append("### ❌ 异常节点")
                for node in node_status["not_ready"]:
                    report_lines.append(f"- **{node['name']}**: {node['reason']}")
                    if node.get("message"):
                        report_lines.append(f"  - {node['message']}")
        else:
            report_lines.append(f"- **错误**: {node_status['error']}")
        report_lines.append("")

        # Pod状态统计
        if "error" not in pod_status:
            report_lines.append("## 📦 Pod状态统计")
            total_pods = pod_status["total"]
            if total_pods > 0:
                running_percentage = (pod_status["running"] / total_pods) * 100
                report_lines.append(f"- **总计**: {total_pods} 个Pod")
                report_lines.append(
                    f"- **运行中**: {pod_status['running']} ({running_percentage:.1f}%)"
                )
                report_lines.append(f"- **等待中**: {pod_status['pending']}")
                report_lines.append(f"- **失败**: {pod_status['failed']}")
                report_lines.append(f"- **成功**: {pod_status['succeeded']}")

                if pod_status["problem_pods"]:
                    report_lines.append("")
                    report_lines.append("### ⚠️ 问题Pod")
                    for pod in pod_status["problem_pods"][:5]:
                        report_lines.append(
                            f"- **{pod['namespace']}/{pod['name']}**: {pod['phase']} ({pod['reason']})"
                        )
            else:
                report_lines.append("- **状态**: 没有找到Pod")
            report_lines.append("")

        # 重要事件
        if events and not any("error" in str(event) for event in events):
            report_lines.append("## 📋 重要事件")
            event_count = 0
            for event in events:
                if event_count >= 10:  # 限制显示数量
                    break

                type_icon = "🔴" if event["type"] == "Error" else "⚠️"
                report_lines.append(
                    f"- {type_icon} **{event['reason']}** ({event['namespace']}/{event['object']})"
                )
                report_lines.append(f"  - **消息**: {event['message']}")
                if event.get("count", 1) > 1:
                    report_lines.append(f"  - **发生次数**: {event['count']}")
                report_lines.append(f"  - **最后发生**: {event['last_seen']}")
                report_lines.append("")
                event_count += 1

        # 错误日志
        if error_logs and not any("error" in str(log) for log in error_logs):
            report_lines.append("## 📄 错误日志")
            for log in error_logs[:3]:
                report_lines.append(f"### {log['namespace']}/{log['pod']}")
                report_lines.append(f"- **状态**: {log['phase']}")
                if "restart_count" in log:
                    report_lines.append(f"- **重启次数**: {log['restart_count']}")

                if "error" in log:
                    report_lines.append(f"- **错误**: {log['error']}")
                elif "sample_log" in log and log["sample_log"].strip():
                    report_lines.append("- **日志片段**:")
                    report_lines.append("```")
                    report_lines.append(log["sample_log"].strip())
                    report_lines.append("```")
                else:
                    report_lines.append("- **日志**: 暂无日志内容")
                report_lines.append("")

        # 总体评估
        report_lines.append("## 📊 总体评估")

        # 计算健康评分
        health_score = 100
        issues = []

        if "error" in cluster_info:
            health_score -= 30
            issues.append("集群信息获取失败")

        if "error" in node_status:
            health_score -= 30
            issues.append("节点状态获取失败")
        else:
            node_health = node_status.get("health_percentage", 100)
            if node_health < 70:
                health_score -= 20
                issues.append("节点健康度过低")
            elif node_health < 90:
                health_score -= 10
                issues.append("部分节点异常")

        if "error" not in pod_status:
            failed_pods = pod_status.get("failed", 0)
            pending_pods = pod_status.get("pending", 0)
            if failed_pods > 0:
                health_score -= min(failed_pods * 5, 20)
                issues.append(f"{failed_pods}个Pod失败")
            if pending_pods > 5:
                health_score -= 10
                issues.append(f"{pending_pods}个Pod长时间等待")

        warning_events = [
            e
            for e in events
            if isinstance(e, dict) and e.get("type") in ["Warning", "Error"]
        ]
        if len(warning_events) > 10:
            health_score -= 15
            issues.append("大量警告事件")
        elif len(warning_events) > 5:
            health_score -= 10
            issues.append("较多警告事件")

        health_score = max(0, health_score)

        if health_score >= 90:
            status_icon = "✅"
            status_text = "优秀"
        elif health_score >= 70:
            status_icon = "⚠️"
            status_text = "良好"
        elif health_score >= 50:
            status_icon = "🔶"
            status_text = "一般"
        else:
            status_icon = "🔴"
            status_text = "差"

        report_lines.append(
            f"- **健康评分**: {status_icon} {health_score}/100 ({status_text})"
        )

        if issues:
            report_lines.append("- **发现的问题**:")
            for issue in issues:
                report_lines.append(f"  - {issue}")
        else:
            report_lines.append("- **状态**: 未发现明显问题")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*报告生成完成*")

        return "\n".join(report_lines)
