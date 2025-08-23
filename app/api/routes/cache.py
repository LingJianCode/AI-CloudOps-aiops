#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 缓存管理API接口
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from app.api.decorators import api_response, log_api_call
from app.common.constants import ErrorMessages, HttpStatusCodes
from app.common.response import ResponseWrapper
from app.core.cache.cache_config import cache_monitor
from app.services.prediction_service import PredictionService
from app.services.rca_service import RCAService

logger = logging.getLogger("aiops.api.cache")

router = APIRouter(tags=["cache"])

# 服务实例
prediction_service = PredictionService()
rca_service = RCAService()


@router.get("/stats", summary="获取缓存统计信息")
@api_response("获取缓存统计信息")
@log_api_call(log_request=False)
async def get_cache_stats() -> Dict[str, Any]:
    """获取系统缓存统计信息"""
    try:
        # 初始化服务以确保缓存管理器可用
        await prediction_service.initialize()
        await rca_service.initialize()
        
        stats = {}
        
        # 获取预测服务缓存统计
        if hasattr(prediction_service, '_cache_manager') and prediction_service._cache_manager:
            pred_stats = prediction_service._cache_manager.get_stats()
            stats['prediction_service'] = {
                'status': 'active',
                'stats': pred_stats,
                'cache_prefix': prediction_service._cache_manager.cache_prefix,
                'default_ttl': prediction_service._cache_manager.default_ttl,
                'max_cache_size': prediction_service._cache_manager.max_cache_size,
            }
        else:
            stats['prediction_service'] = {'status': 'inactive', 'reason': 'cache_manager_not_initialized'}
        
        # 获取RCA服务缓存统计
        if hasattr(rca_service, '_cache_manager') and rca_service._cache_manager:
            rca_stats = rca_service._cache_manager.get_stats()
            stats['rca_service'] = {
                'status': 'active',
                'stats': rca_stats,
                'cache_prefix': rca_service._cache_manager.cache_prefix,
                'default_ttl': rca_service._cache_manager.default_ttl,
                'max_cache_size': rca_service._cache_manager.max_cache_size,
            }
        else:
            stats['rca_service'] = {'status': 'inactive', 'reason': 'cache_manager_not_initialized'}
        
        # 获取全局缓存监控统计
        monitor_stats = cache_monitor.get_cache_stats()
        performance_insights = cache_monitor.get_performance_insights()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'service_stats': stats,
            'monitor_stats': monitor_stats,
            'performance_insights': performance_insights,
            'overall_status': 'healthy' if any(s.get('status') == 'active' for s in stats.values()) else 'unhealthy'
        }
        
        return ResponseWrapper.success(data=result, message="缓存统计信息获取成功")
        
    except Exception as e:
        logger.error(f"获取缓存统计失败: {str(e)}")
        raise HTTPException(
            status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
            detail=f"获取缓存统计失败: {str(e)}"
        )


@router.get("/health", summary="缓存系统健康检查")
@api_response("缓存系统健康检查")
@log_api_call(log_request=False)
async def cache_health_check() -> Dict[str, Any]:
    """检查缓存系统健康状态"""
    try:
        health_info = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'services': {}
        }
        
        services_healthy = 0
        total_services = 0
        
        # 检查预测服务缓存
        try:
            await prediction_service.initialize()
            if hasattr(prediction_service, '_cache_manager') and prediction_service._cache_manager:
                pred_health = prediction_service._cache_manager.health_check()
                health_info['services']['prediction'] = pred_health
                if pred_health.get('status') == 'healthy':
                    services_healthy += 1
            else:
                health_info['services']['prediction'] = {
                    'status': 'unavailable',
                    'redis_connected': False,
                    'error': 'cache_manager_not_initialized'
                }
            total_services += 1
        except Exception as e:
            health_info['services']['prediction'] = {
                'status': 'error',
                'redis_connected': False,
                'error': str(e)
            }
            total_services += 1
        
        # 检查RCA服务缓存
        try:
            await rca_service.initialize()
            if hasattr(rca_service, '_cache_manager') and rca_service._cache_manager:
                rca_health = rca_service._cache_manager.health_check()
                health_info['services']['rca'] = rca_health
                if rca_health.get('status') == 'healthy':
                    services_healthy += 1
            else:
                health_info['services']['rca'] = {
                    'status': 'unavailable',
                    'redis_connected': False,
                    'error': 'cache_manager_not_initialized'
                }
            total_services += 1
        except Exception as e:
            health_info['services']['rca'] = {
                'status': 'error',
                'redis_connected': False,
                'error': str(e)
            }
            total_services += 1
        
        # 确定总体健康状态
        if services_healthy == total_services:
            health_info['overall_status'] = 'healthy'
        elif services_healthy > 0:
            health_info['overall_status'] = 'degraded'
        else:
            health_info['overall_status'] = 'unhealthy'
        
        health_info['summary'] = {
            'healthy_services': services_healthy,
            'total_services': total_services,
            'health_percentage': round(services_healthy / total_services * 100, 1) if total_services > 0 else 0
        }
        
        return ResponseWrapper.success(data=health_info, message="缓存健康检查完成")
        
    except Exception as e:
        logger.error(f"缓存健康检查失败: {str(e)}")
        raise HTTPException(
            status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
            detail=f"缓存健康检查失败: {str(e)}"
        )


@router.post("/clear", summary="清空缓存")
@api_response("清空缓存")
@log_api_call(log_request=True)
async def clear_cache(
    service: str = Query(..., description="服务名称: prediction, rca, 或 all"),
    pattern: str = Query(None, description="可选的模式匹配，用于部分清空")
) -> Dict[str, Any]:
    """清空指定服务的缓存"""
    try:
        results = {}
        
        if service in ['prediction', 'all']:
            await prediction_service.initialize()
            if hasattr(prediction_service, '_cache_manager') and prediction_service._cache_manager:
                if pattern:
                    result = prediction_service._cache_manager.clear_pattern(pattern)
                else:
                    result = prediction_service._cache_manager.clear_all()
                results['prediction'] = result
            else:
                results['prediction'] = {'success': False, 'message': 'cache_manager_not_available'}
        
        if service in ['rca', 'all']:
            await rca_service.initialize()
            if hasattr(rca_service, '_cache_manager') and rca_service._cache_manager:
                if pattern:
                    result = rca_service._cache_manager.clear_pattern(pattern)
                else:
                    result = rca_service._cache_manager.clear_all()
                results['rca'] = result
            else:
                results['rca'] = {'success': False, 'message': 'cache_manager_not_available'}
        
        if service not in ['prediction', 'rca', 'all']:
            raise HTTPException(
                status_code=HttpStatusCodes.BAD_REQUEST,
                detail="service参数必须是: prediction, rca, 或 all"
            )
        
        # 计算总体结果
        total_cleared = sum(r.get('cleared_count', 0) for r in results.values())
        all_successful = all(r.get('success', False) for r in results.values())
        
        response_data = {
            'timestamp': datetime.now().isoformat(),
            'service': service,
            'pattern': pattern,
            'results': results,
            'summary': {
                'total_cleared': total_cleared,
                'all_successful': all_successful
            }
        }
        
        message = f"缓存清空完成，共清除 {total_cleared} 条记录"
        if pattern:
            message += f"（模式: {pattern}）"
        
        return ResponseWrapper.success(data=response_data, message=message)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清空缓存失败: {str(e)}")
        raise HTTPException(
            status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
            detail=f"清空缓存失败: {str(e)}"
        )


@router.get("/performance", summary="获取缓存性能报告")
@api_response("获取缓存性能报告")
@log_api_call(log_request=False)
async def get_cache_performance() -> Dict[str, Any]:
    """获取详细的缓存性能报告"""
    try:
        # 获取性能洞察
        performance_insights = cache_monitor.get_performance_insights()
        cache_stats = cache_monitor.get_cache_stats()
        
        # 计算整体指标
        total_requests = sum(stats.get('total_requests', 0) for stats in cache_stats.values())
        total_hits = sum(stats.get('cache_hits', 0) for stats in cache_stats.values())
        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        # 性能等级评估
        if overall_hit_rate >= 80:
            performance_grade = 'A'
            performance_desc = '优秀'
        elif overall_hit_rate >= 60:
            performance_grade = 'B'
            performance_desc = '良好'
        elif overall_hit_rate >= 40:
            performance_grade = 'C'
            performance_desc = '一般'
        else:
            performance_grade = 'D'
            performance_desc = '需要优化'
        
        # 生成建议
        recommendations = []
        for cache_type, stats in cache_stats.items():
            hit_rate = stats.get('hit_rate', 0)
            if hit_rate < 50:
                recommendations.append(f"{cache_type}: 命中率过低({hit_rate:.1f}%)，建议检查缓存键生成逻辑")
            elif hit_rate < 70:
                recommendations.append(f"{cache_type}: 命中率偏低({hit_rate:.1f}%)，建议调整TTL配置")
        
        if not recommendations:
            recommendations.append("缓存性能良好，无需特别优化")
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': {
                'total_requests': total_requests,
                'total_hits': total_hits,
                'overall_hit_rate': round(overall_hit_rate, 2),
                'performance_grade': performance_grade,
                'performance_description': performance_desc
            },
            'performance_insights': performance_insights,
            'cache_stats_by_type': cache_stats,
            'recommendations': recommendations,
            'cache_effectiveness': performance_insights.get('cache_effectiveness', 'unknown')
        }
        
        return ResponseWrapper.success(data=result, message="缓存性能报告生成成功")
        
    except Exception as e:
        logger.error(f"获取缓存性能报告失败: {str(e)}")
        raise HTTPException(
            status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
            detail=f"获取缓存性能报告失败: {str(e)}"
        )


@router.get("/config", summary="获取缓存配置信息")
@api_response("获取缓存配置信息")
@log_api_call(log_request=False)
async def get_cache_config() -> Dict[str, Any]:
    """获取当前缓存配置信息"""
    try:
        from app.core.cache.cache_config import CacheStrategy, CacheType
        
        config_info = {
            'timestamp': datetime.now().isoformat(),
            'cache_types': [cache_type.value for cache_type in CacheType],
            'ttl_config': {cache_type.value: CacheStrategy.CACHE_TTL_CONFIG[cache_type] 
                          for cache_type in CacheType},
            'priority_config': {cache_type.value: CacheStrategy.CACHE_PRIORITY_CONFIG[cache_type] 
                               for cache_type in CacheType},
            'prefix_config': {cache_type.value: CacheStrategy.CACHE_PREFIX_CONFIG[cache_type] 
                             for cache_type in CacheType},
            'compression_thresholds': {
                cache_type.value: CacheStrategy.get_cache_compression_threshold(cache_type)
                for cache_type in CacheType
            }
        }
        
        # 添加服务配置信息
        service_configs = {}
        
        try:
            await prediction_service.initialize()
            if hasattr(prediction_service, '_cache_manager') and prediction_service._cache_manager:
                service_configs['prediction'] = {
                    'cache_prefix': prediction_service._cache_manager.cache_prefix,
                    'default_ttl': prediction_service._cache_manager.default_ttl,
                    'max_cache_size': prediction_service._cache_manager.max_cache_size,
                    'enable_compression': prediction_service._cache_manager.enable_compression
                }
        except:
            service_configs['prediction'] = {'status': 'not_available'}
        
        try:
            await rca_service.initialize()
            if hasattr(rca_service, '_cache_manager') and rca_service._cache_manager:
                service_configs['rca'] = {
                    'cache_prefix': rca_service._cache_manager.cache_prefix,
                    'default_ttl': rca_service._cache_manager.default_ttl,
                    'max_cache_size': rca_service._cache_manager.max_cache_size,
                    'enable_compression': rca_service._cache_manager.enable_compression
                }
        except:
            service_configs['rca'] = {'status': 'not_available'}
        
        config_info['service_configs'] = service_configs
        
        return ResponseWrapper.success(data=config_info, message="缓存配置信息获取成功")
        
    except Exception as e:
        logger.error(f"获取缓存配置失败: {str(e)}")
        raise HTTPException(
            status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
            detail=f"获取缓存配置失败: {str(e)}"
        )


__all__ = ["router"]
