#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: AIOps平台全量测试脚本，运行所有测试模块
"""

import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import pytest

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_all")


def ensure_logs_directory():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def run_all_tests():
    start_time = time.time()
    logger.info("开始运行AI-CloudOps平台完整测试套件...")
    
    # 确保日志目录存在
    logs_dir = ensure_logs_directory()
    
    # 获取测试目录
    test_dir = Path(__file__).parent
    
    # 定义测试模块顺序（按重要性和依赖关系排序）
    test_modules = [
        "test_health.py",          # 健康检查（最基础）
        "test_predict.py",         # 负载预测
        "test_rca.py",            # 根因分析  
        "test_autofix.py",        # 自动修复
        "test_assistant.py",      # 智能助手
        "test_integration.py",    # 系统集成测试
    ]
    
    # 收集结果
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "test_suite": "AI-CloudOps完整测试套件",
        "modules": {},
        "summary": {}
    }
    
    successful_modules = 0
    total_modules = len(test_modules)
    
    # 运行测试
    for i, module in enumerate(test_modules, 1):
        module_path = test_dir / module
        
        if not module_path.exists():
            logger.warning(f"测试模块不存在: {module}")
            results["modules"][module] = {
                "status": "跳过",
                "reason": "文件不存在",
                "exit_code": None
            }
            continue
        
        logger.info(f"[{i}/{total_modules}] 运行测试模块: {module}")
        module_start_time = time.time()
        
        # 运行pytest
        try:
            exit_code = pytest.main([
                "-v",                    # 详细输出
                "--tb=short",           # 简短的traceback
                "--disable-warnings",   # 禁用警告
                str(module_path)
            ])
            
            module_duration = time.time() - module_start_time
            
            if exit_code == 0:
                status = "通过"
                successful_modules += 1
                logger.info(f"✅ {module} 测试通过 (耗时: {module_duration:.2f}秒)")
            else:
                status = "失败"
                logger.error(f"❌ {module} 测试失败 (耗时: {module_duration:.2f}秒)")
            
            results["modules"][module] = {
                "status": status,
                "exit_code": exit_code,
                "duration": module_duration,
                "success": exit_code == 0
            }
            
        except Exception as e:
            module_duration = time.time() - module_start_time
            logger.error(f"💥 {module} 执行异常: {str(e)} (耗时: {module_duration:.2f}秒)")
            results["modules"][module] = {
                "status": "异常",
                "exit_code": -1,
                "duration": module_duration,
                "error": str(e),
                "success": False
            }
    
    # 计算总结
    total_duration = time.time() - start_time
    success_rate = (successful_modules / total_modules) * 100 if total_modules > 0 else 0
    
    results["summary"] = {
        "total_modules": total_modules,
        "successful_modules": successful_modules,
        "failed_modules": total_modules - successful_modules,
        "success_rate": f"{success_rate:.1f}%",
        "total_duration": total_duration,
        "overall_success": successful_modules == total_modules
    }
    
    # 输出结果摘要
    print_test_summary(results)
    
    # 保存详细结果
    save_test_results(results, logs_dir)
    
    return successful_modules == total_modules


def print_test_summary(results):
    """打印测试摘要"""
    summary = results["summary"]
    
    print("\n" + "=" * 80)
    print(" AI-CloudOps平台测试结果摘要")
    print("=" * 80)
    print(f"总测试模块数: {summary['total_modules']}")
    print(f"成功模块数: {summary['successful_modules']}")
    print(f"失败模块数: {summary['failed_modules']}")
    print(f"成功率: {summary['success_rate']}")
    print(f"总耗时: {summary['total_duration']:.2f} 秒")
    print(f"测试时间: {results['timestamp']}")
    
    print("\n详细模块结果:")
    print("-" * 80)
    
    for module, result in results["modules"].items():
        if result["status"] == "通过":
            icon = "✅"
        elif result["status"] == "失败": 
            icon = "❌"
        elif result["status"] == "异常":
            icon = "💥"
        else:
            icon = "⏭️"
        
        duration = result.get("duration", 0)
        print(f"{icon} {module:<25} {result['status']:<6} ({duration:.2f}秒)")
    
    print("=" * 80)
    
    # 整体结果
    if summary["overall_success"]:
        print("🎉 所有测试模块均通过！系统状态良好。")
    else:
        print("⚠️  部分测试模块失败，请检查具体错误信息。")
    print("=" * 80)


def save_test_results(results, logs_dir):
    """保存测试结果到文件"""
    try:
        import json
        
        # 保存JSON格式结果
        result_file = logs_dir / f"test_all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"测试结果已保存到: {result_file}")
        
        # 保存简化的文本报告
        report_file = logs_dir / f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("AI-CloudOps平台测试报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"测试时间: {results['timestamp']}\n")
            f.write(f"成功率: {results['summary']['success_rate']}\n")
            f.write(f"总耗时: {results['summary']['total_duration']:.2f}秒\n\n")
            
            f.write("模块详情:\n")
            f.write("-" * 30 + "\n")
            for module, result in results["modules"].items():
                f.write(f"{module}: {result['status']}\n")
        
        logger.info(f"测试报告已保存到: {report_file}")
        
    except Exception as e:
        logger.error(f"保存测试结果失败: {str(e)}")


def main():
    """主函数"""
    try:
        logger.info("AI-CloudOps平台测试套件启动")
        success = run_all_tests()
        exit_code = 0 if success else 1
        
        if success:
            logger.info("🎉 所有测试完成，系统状态良好！")
        else:
            logger.warning("⚠️ 部分测试失败，请查看详细报告")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        sys.exit(130)
    except Exception as e:
        logger.error(f"测试过程出现未处理异常: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()