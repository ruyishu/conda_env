#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将GSM8K数据集转换为LLaMA-Factory所需的Alpaca格式
"""

import json
import pandas as pd
from pathlib import Path

def convert_gsm8k_to_alpaca(input_parquet_path: str, output_json_path: str, max_samples: int = None):
    """
    将GSM8K parquet文件转换为Alpaca JSON格式
    
    Args:
        input_parquet_path: 输入的parquet文件路径
        output_json_path: 输出的JSON文件路径
        max_samples: 最大样本数量，None表示使用全部数据
    """
    print(f"Loading GSM8K data from {input_parquet_path}")
    df = pd.read_parquet(input_parquet_path)
    
    if max_samples is not None:
        df = df.head(max_samples)
        print(f"Using first {max_samples} samples")
    else:
        print(f"Using all {len(df)} samples")
    
    # 转换为Alpaca格式
    alpaca_data = []
    for _, row in df.iterrows():
        question = str(row['question']).strip()
        answer = str(row['answer']).strip()
        
        alpaca_sample = {
            "instruction": "请解决这个数学问题，给出详细的解题步骤和最终答案。",
            "input": question,
            "output": answer,
            "system": "你是一个数学专家，擅长解决各种数学问题。"
        }
        alpaca_data.append(alpaca_sample)
    
    # 保存为JSON文件
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(alpaca_data)} samples to Alpaca format")
    print(f"Saved to {output_json_path}")
    
    # 显示示例
    if alpaca_data:
        print("\n示例数据:")
        print(json.dumps(alpaca_data[0], ensure_ascii=False, indent=2))

def main():
    # 配置路径
    input_train_path = "/zk/data/train-00000-of-00001.parquet"
    input_test_path = "/zk/data/test-00000-of-00001.parquet"
    
    output_train_path = "/zk/code/llamafactorysft/data/gsm8k_train_alpaca.json"
    output_test_path = "/zk/code/llamafactorysft/data/gsm8k_test_alpaca.json"
    
    # 转换训练集（使用前1000条进行快速测试）
    print("=" * 60)
    print("转换训练集")
    print("=" * 60)
    convert_gsm8k_to_alpaca(input_train_path, output_train_path, max_samples=1000)
    
    # 转换测试集（使用前100条）
    print("\n" + "=" * 60)
    print("转换测试集")
    print("=" * 60)
    convert_gsm8k_to_alpaca(input_test_path, output_test_path, max_samples=100)
    
    print("\n数据转换完成！")

if __name__ == "__main__":
    main()