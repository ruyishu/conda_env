# LLaMA-Factory SFT微调完整流程指南

本文档提供了使用LLaMA-Factory进行SFT（Supervised Fine-Tuning）微调的完整流程，适合初学者按步骤操作。

## 目录
1. [环境准备](#环境准备)
2. [项目结构](#项目结构)
3. [数据准备](#数据准备)
4. [配置文件](#配置文件)
5. [开始训练](#开始训练)
6. [模型合并](#模型合并)
7. [常见问题](#常见问题)

## 环境准备

### 1. 安装LLaMA-Factory
```bash
# 克隆LLaMA-Factory仓库
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装依赖
pip install -e .
```

### 2. 验证安装
```bash
# 检查是否安装成功
llamafactory-cli --help
```

## 项目结构

创建以下目录结构：
```
your_project/
├── data/                    # 数据文件夹
│   ├── dataset_info.json    # 数据集配置文件
│   └── your_data.json       # 训练数据
├── configs/                 # 配置文件夹
│   ├── train_config.yaml    # 训练配置
│   └── merge_config.yaml    # 合并配置
├── saves/                   # 模型保存文件夹
├── logs/                    # 日志文件夹
└── models/                  # 合并后模型文件夹
```

## 数据准备

### 1. 数据格式

LLaMA-Factory支持多种数据格式，推荐使用Alpaca格式：

```json
[
  {
    "instruction": "请解决这个数学问题，给出详细的解题步骤和最终答案。",
    "input": "小明有5个苹果，吃了2个，还剩几个？",
    "output": "小明原来有5个苹果，吃了2个苹果。\n5 - 2 = 3\n所以小明还剩3个苹果。",
    "system": "你是一个数学专家，擅长解决各种数学问题。"
  }
]
```

### 2. 数据转换脚本示例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据转换脚本示例
"""

import json
import pandas as pd
from pathlib import Path

def convert_to_alpaca(input_file: str, output_file: str, max_samples: int = None):
    """
    将原始数据转换为Alpaca格式
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        max_samples: 最大样本数量，None表示使用全部数据
    """
    # 根据输入文件格式选择读取方式
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.json'):
        df = pd.read_json(input_file)
    else:
        raise ValueError(f"不支持的文件格式: {input_file}")
    
    if max_samples is not None:
        df = df.head(max_samples)
        print(f"使用前 {max_samples} 条样本")
    else:
        print(f"使用全部 {len(df)} 条样本")
    
    # 转换为Alpaca格式
    alpaca_data = []
    for _, row in df.iterrows():
        # 根据你的数据字段调整
        question = str(row['question']).strip()  # 替换为你的问题字段
        answer = str(row['answer']).strip()      # 替换为你的答案字段
        
        alpaca_sample = {
            "instruction": "请根据问题给出准确的回答。",  # 自定义指令
            "input": question,
            "output": answer,
            "system": "你是一个有用的AI助手。"  # 自定义系统提示
        }
        alpaca_data.append(alpaca_sample)
    
    # 保存为JSON文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成，共 {len(alpaca_data)} 条样本")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    # 示例用法
    convert_to_alpaca(
        input_file="your_input_data.parquet",
        output_file="data/your_train_data.json",
        max_samples=1000  # 可选：限制样本数量
    )
```

### 3. 配置dataset_info.json

在`data/dataset_info.json`中注册你的数据集：

```json
{
  "your_dataset_train": {
    "file_name": "your_train_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    },
    "tags": {
      "language": "zh",
      "task": "general"
    }
  },
  "your_dataset_test": {
    "file_name": "your_test_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    },
    "tags": {
      "language": "zh",
      "task": "general"
    }
  }
}
```

## 配置文件

### 1. 训练配置文件 (configs/train_config.yaml)

```yaml
### 模型配置
model_name_or_path: /path/to/your/base/model  # 基础模型路径

### 训练方法
stage: sft                    # 训练阶段：sft
do_train: true               # 启用训练
finetuning_type: lora        # 微调类型：lora
lora_target: all             # LoRA目标层：all表示所有线性层
lora_rank: 8                 # LoRA秩：通常4-64
lora_alpha: 16               # LoRA alpha：通常是rank的2倍
lora_dropout: 0.1            # LoRA dropout：0.05-0.1

### 数据集配置
dataset_dir: ./data          # 数据集目录
dataset: your_dataset_train  # 数据集名称（对应dataset_info.json中的key）
template: qwen               # 模板类型：根据模型选择（qwen/llama/chatglm等）
cutoff_len: 2048            # 最大序列长度
max_samples: 1000           # 最大样本数（可选，用于快速测试）
overwrite_cache: true       # 覆盖缓存
preprocessing_num_workers: 16 # 预处理工作进程数

### 输出配置
output_dir: ./saves/your_model_lora  # 输出目录
logging_steps: 10           # 日志记录步数
save_steps: 200             # 模型保存步数
plot_loss: true             # 绘制损失曲线
overwrite_output_dir: true  # 覆盖输出目录

### 训练参数
per_device_train_batch_size: 2    # 每设备训练批次大小
gradient_accumulation_steps: 4    # 梯度累积步数
learning_rate: 1.0e-4            # 学习率
num_train_epochs: 3.0            # 训练轮数
lr_scheduler_type: cosine        # 学习率调度器
warmup_ratio: 0.1               # 预热比例
bf16: true                      # 使用bf16精度（需要支持）
ddp_timeout: 180000000          # DDP超时时间

### 评估配置
val_size: 0.1                   # 验证集比例
per_device_eval_batch_size: 1   # 每设备评估批次大小
eval_strategy: steps            # 评估策略
eval_steps: 100                 # 评估步数

### 保存配置
save_only_model: false          # 是否只保存模型
save_total_limit: 3             # 保存检查点数量限制

### 日志配置
logging_dir: ./logs             # 日志目录
```

### 2. 合并配置文件 (configs/merge_config.yaml)

```yaml
### 模型配置
model_name_or_path: /path/to/your/base/model     # 基础模型路径
adapter_name_or_path: ./saves/your_model_lora    # LoRA适配器路径
template: qwen                                   # 模板类型
finetuning_type: lora                           # 微调类型

### 导出配置
export_dir: ./models/your_merged_model          # 合并后模型保存路径
export_size: 2                                  # 导出分片大小（GB）
export_device: cpu                              # 导出设备
export_legacy_format: false                    # 是否使用旧格式
```

## 开始训练

### 1. 启动训练

```bash
# 方法1：使用命令行
llamafactory-cli train configs/train_config.yaml

# 方法2：使用Python脚本
python -m llamafactory.train configs/train_config.yaml
```

### 2. 监控训练

```bash
# 查看训练日志
tail -f logs/train.log

# 使用TensorBoard监控（如果启用了logging_dir）
tensorboard --logdir=./logs
```

### 3. 训练参数调优建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| lora_rank | 4-64 | 越大模型容量越大，但训练越慢 |
| lora_alpha | rank*2 | 通常设为rank的2倍 |
| learning_rate | 1e-4 到 5e-4 | 根据数据集大小调整 |
| batch_size | 2-8 | 根据显存大小调整 |
| num_train_epochs | 1-5 | 根据数据集大小调整 |

## 模型合并

训练完成后，需要将LoRA权重与基础模型合并：

```bash
# 合并模型
llamafactory-cli export configs/merge_config.yaml
```

合并后的模型保存在`export_dir`指定的目录中，可以直接使用。

## 常见问题

### 1. 显存不足
- 减小`per_device_train_batch_size`
- 增加`gradient_accumulation_steps`
- 使用`fp16`或`bf16`
- 减小`cutoff_len`

### 2. 训练速度慢
- 增加`per_device_train_batch_size`
- 减少`gradient_accumulation_steps`
- 增加`preprocessing_num_workers`
- 使用更快的存储设备

### 3. 模型效果不好
- 增加训练数据量
- 调整学习率
- 增加训练轮数
- 检查数据质量

### 4. 常见错误

#### 错误：`modules_to_save` not supported
**解决方案**：删除配置文件中的`modules_to_save`参数

#### 错误：Dataset not found
**解决方案**：检查`dataset_info.json`中的数据集名称和文件路径

#### 错误：Template not found
**解决方案**：确认模板名称正确，常见模板：`qwen`、`llama`、`chatglm`

### 5. 性能优化

```yaml
# 针对不同显存的推荐配置

# 8GB显存
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
cutoff_len: 1024
lora_rank: 8

# 16GB显存
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
cutoff_len: 2048
lora_rank: 16

# 24GB显存及以上
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
cutoff_len: 4096
lora_rank: 32
```

## 完整示例脚本

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA-Factory SFT训练完整流程脚本
"""

import os
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sft_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_command(command, cwd=None):
    """执行命令"""
    logger = logging.getLogger(__name__)
    logger.info(f"执行命令: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"命令执行成功: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e.stderr}")
        raise

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始SFT训练流程")
    
    # 1. 检查LLaMA-Factory是否安装
    try:
        run_command("llamafactory-cli --help")
        logger.info("LLaMA-Factory已安装")
    except:
        logger.error("LLaMA-Factory未安装，请先安装")
        return
    
    # 2. 开始训练
    logger.info("开始训练...")
    train_config = "configs/train_config.yaml"
    if not Path(train_config).exists():
        logger.error(f"训练配置文件不存在: {train_config}")
        return
    
    run_command(f"llamafactory-cli train {train_config}")
    logger.info("训练完成")
    
    # 3. 合并模型
    logger.info("开始合并模型...")
    merge_config = "configs/merge_config.yaml"
    if not Path(merge_config).exists():
        logger.error(f"合并配置文件不存在: {merge_config}")
        return
    
    run_command(f"llamafactory-cli export {merge_config}")
    logger.info("模型合并完成")
    
    logger.info("SFT训练流程全部完成！")

if __name__ == "__main__":
    main()
```

## 总结

按照以上流程，你可以轻松完成LLaMA-Factory的SFT微调：

1. **准备环境**：安装LLaMA-Factory
2. **准备数据**：转换为Alpaca格式，配置dataset_info.json
3. **配置训练**：编写train_config.yaml
4. **开始训练**：使用llamafactory-cli train
5. **合并模型**：使用llamafactory-cli export

记住根据你的具体需求调整配置参数，特别是模型路径、数据集名称和训练参数。

---

*本文档基于LLaMA-Factory实际使用经验编写，如有问题请参考官方文档或提交Issue。*