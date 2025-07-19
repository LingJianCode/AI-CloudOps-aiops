#!/usr/bin/env python3
"""
K8s测试环境设置脚本
创建有问题的部署用于自动修复测试
"""

import subprocess
import time
import yaml
import os

# 测试部署配置
test_deployments = {
    'nginx-probe-error': {
        'replicas': 1,
        'image': 'nginx:1.20-alpine',
        'liveness_probe': {
            'httpGet': {'path': '/nonexistent', 'port': 80},
            'initialDelaySeconds': 5,
            'periodSeconds': 5
        },
        'resources': {
            'requests': {'memory': '32Mi', 'cpu': '25m'},
            'limits': {'memory': '64Mi', 'cpu': '50m'}
        }
    },
    'app-no-health-checks': {
        'replicas': 1,
        'image': 'nginx:1.20-alpine',
        'command': ['/bin/sh', '-c', 'sleep 10 && nginx -g "daemon off;"'],
        'resources': {
            'requests': {'memory': '32Mi', 'cpu': '25m'},
            'limits': {'memory': '64Mi', 'cpu': '50m'}
        }
    },
    'app-high-resources': {
        'replicas': 1,
        'image': 'nginx:1.20-alpine',
        'resources': {
            'requests': {'memory': '512Mi', 'cpu': '500m'},
            'limits': {'memory': '1Gi', 'cpu': '1000m'}
        }
    }
}

def run_command(cmd):
    """执行shell命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def create_deployment(name, config):
    """创建K8s部署"""
    deployment = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': name,
            'namespace': 'aiops-testing',
            'labels': {'app': name}
        },
        'spec': {
            'replicas': config['replicas'],
            'selector': {
                'matchLabels': {'app': name}
            },
            'template': {
                'metadata': {
                    'labels': {'app': name}
                },
                'spec': {
                    'containers': [{
                        'name': 'main',
                        'image': config['image'],
                        'ports': [{'containerPort': 80}],
                        'resources': {
                            'requests': config['resources']['requests'],
                            'limits': config['resources']['limits']
                        }
                    }]
                }
            }
        }
    }
    
    # 添加探针配置
    if 'liveness_probe' in config:
        deployment['spec']['template']['spec']['containers'][0]['livenessProbe'] = config['liveness_probe']
    
    # 添加启动命令
    if 'command' in config:
        deployment['spec']['template']['spec']['containers'][0]['command'] = config['command']
    
    return deployment

def setup_test_environment():
    """设置测试环境"""
    print("🚀 开始设置K8s测试环境...")
    
    # 创建命名空间
    success, stdout, stderr = run_command("kubectl create namespace aiops-testing --dry-run=client -o yaml | kubectl apply -f -")
    if success:
        print("✅ 命名空间 aiops-testing 已创建")
    else:
        print(f"❌ 创建命名空间失败: {stderr}")
    
    # 创建部署YAML文件
    yaml_dir = "data/sample/generated"
    os.makedirs(yaml_dir, exist_ok=True)
    
    for name, config in test_deployments.items():
        deployment = create_deployment(name, config)
        yaml_file = f"{yaml_dir}/{name}.yaml"
        
        with open(yaml_file, 'w') as f:
            yaml.dump(deployment, f, default_flow_style=False)
        
        # 应用部署
        success, stdout, stderr = run_command(f"kubectl apply -f {yaml_file}")
        if success:
            print(f"✅ 部署 {name} 已创建")
        else:
            print(f"❌ 部署 {name} 失败: {stderr}")
    
    print("⏳ 等待部署启动...")
    time.sleep(10)
    
    # 检查状态
    success, stdout, stderr = run_command("kubectl get pods -n aiops-testing")
    if success:
        print("📊 当前Pod状态:")
        print(stdout)
    
    print("🎯 测试环境设置完成！")
    print("可用的测试部署:")
    for name in test_deployments.keys():
        print(f"  - {name}")

if __name__ == "__main__":
    setup_test_environment()