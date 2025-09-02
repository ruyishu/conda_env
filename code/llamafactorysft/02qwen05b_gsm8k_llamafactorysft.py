#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Qwen2.5-0.5B GSM8K SFT Training Script using LLaMA-Factory
=============================================================================

【代码功能概述】
使用 LLaMA-Factory 对 Qwen2.5-0.5B-Instruct 在 GSM8K 数据集上做监督微调（SFT），
采用 LoRA 技术进行高效微调。

【主要功能】
1. 自动安装和配置 LLaMA-Factory
2. 数据集准备和格式转换
3. 启动 LoRA SFT 训练
4. 模型合并和保存
5. 训练监控和日志记录

【使用方法】
1. 确保已准备好基础模型和数据集
2. 运行脚本：python 02qwen05b_gsm8k_llamafactorysft.py
3. 监控训练：tensorboard --logdir=./logs

【依赖】
pip install torch transformers datasets accelerate peft bitsandbytes tensorboard
git clone https://github.com/hiyouga/LLaMA-Factory.git

=============================================================================
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# 配置参数
# =============================================================================

# 项目路径配置
PROJECT_ROOT = Path("/zk/code/llamafactorysft")
LLAMAFACTORY_DIR = PROJECT_ROOT / "LLaMA-Factory"
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
SAVES_DIR = PROJECT_ROOT / "saves"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# 训练配置
TRAIN_CONFIG_FILE = "qwen05b_lora_sft.yaml"
MERGE_CONFIG_FILE = "qwen05b_merge.yaml"

# 模型和数据路径
BASE_MODEL_PATH = "/zk/Qwen2.5-0.5B-Instruct"
TRAIN_DATA_PATH = DATA_DIR / "gsm8k_train_alpaca.json"
TEST_DATA_PATH = DATA_DIR / "gsm8k_test_alpaca.json"

# 全局 logger
logger: Optional[logging.Logger] = None

# =============================================================================
# 工具函数
# =============================================================================

def setup_logging() -> logging.Logger:
    """设置日志配置"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"llamafactory_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_command(cmd: str, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """运行命令并返回结果"""
    logger.info(f"Running command: {cmd}")
    if cwd:
        logger.info(f"Working directory: {cwd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        
        if result.stdout:
            logger.info(f"STDOUT: {result.stdout}")
        if result.stderr:
            logger.warning(f"STDERR: {result.stderr}")
            
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise

def check_prerequisites():
    """检查前置条件"""
    logger.info("Checking prerequisites...")
    
    # 检查基础模型
    if not Path(BASE_MODEL_PATH).exists():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL_PATH}")
    logger.info(f"✓ Base model found: {BASE_MODEL_PATH}")
    
    # 检查数据集
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA_PATH}")
    logger.info(f"✓ Training data found: {TRAIN_DATA_PATH}")
    
    # 检查配置文件
    config_path = CONFIG_DIR / TRAIN_CONFIG_FILE
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    logger.info(f"✓ Training config found: {config_path}")

def install_llamafactory():
    """安装 LLaMA-Factory"""
    logger.info("Installing LLaMA-Factory...")
    
    if LLAMAFACTORY_DIR.exists():
        logger.info("LLaMA-Factory already exists, skipping installation")
        return
    
    # 克隆仓库
    clone_cmd = f"git clone https://github.com/hiyouga/LLaMA-Factory.git {LLAMAFACTORY_DIR}"
    run_command(clone_cmd, cwd=PROJECT_ROOT)
    
    # 安装依赖
    install_cmd = 'pip install -e ".[torch,metrics]"'
    run_command(install_cmd, cwd=LLAMAFACTORY_DIR)
    
    logger.info("✓ LLaMA-Factory installed successfully")

def prepare_llamafactory_config():
    """准备 LLaMA-Factory 配置"""
    logger.info("Preparing LLaMA-Factory configuration...")
    
    # 复制数据集配置到 LLaMA-Factory
    src_dataset_info = DATA_DIR / "dataset_info.json"
    dst_dataset_info = LLAMAFACTORY_DIR / "data" / "dataset_info.json"
    
    # 读取现有配置（如果存在）
    existing_config = {}
    if dst_dataset_info.exists():
        with open(dst_dataset_info, 'r', encoding='utf-8') as f:
            existing_config = json.load(f)
    
    # 读取我们的配置
    with open(src_dataset_info, 'r', encoding='utf-8') as f:
        our_config = json.load(f)
    
    # 合并配置
    existing_config.update(our_config)
    
    # 保存合并后的配置
    with open(dst_dataset_info, 'w', encoding='utf-8') as f:
        json.dump(existing_config, f, ensure_ascii=False, indent=2)
    
    # 复制数据文件到 LLaMA-Factory
    llamafactory_data_dir = LLAMAFACTORY_DIR / "data"
    llamafactory_data_dir.mkdir(exist_ok=True)
    
    import shutil
    shutil.copy2(TRAIN_DATA_PATH, llamafactory_data_dir / TRAIN_DATA_PATH.name)
    shutil.copy2(TEST_DATA_PATH, llamafactory_data_dir / TEST_DATA_PATH.name)
    
    logger.info("✓ LLaMA-Factory configuration prepared")

def start_training():
    """启动训练"""
    logger.info("Starting SFT training...")
    
    config_path = CONFIG_DIR / TRAIN_CONFIG_FILE
    
    # 构建训练命令
    train_cmd = f"llamafactory-cli train {config_path}"
    
    logger.info("=" * 60)
    logger.info("TRAINING STARTED")
    logger.info("=" * 60)
    logger.info(f"Config file: {config_path}")
    logger.info(f"Output directory: {SAVES_DIR / 'qwen05b_lora_sft'}")
    logger.info(f"Logs directory: {LOGS_DIR}")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # 运行训练（不捕获输出，让其实时显示）
        result = subprocess.run(
            train_cmd,
            shell=True,
            cwd=LLAMAFACTORY_DIR,
            check=True
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Training duration: {training_duration}")
        logger.info(f"Model saved to: {SAVES_DIR / 'qwen05b_lora_sft'}")
        logger.info("=" * 60)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        return False

def create_merge_config():
    """创建模型合并配置"""
    logger.info("Creating merge configuration...")
    
    merge_config = {
        "model_name_or_path": BASE_MODEL_PATH,
        "adapter_name_or_path": str(SAVES_DIR / "qwen05b_lora_sft"),
        "template": "qwen",
        "finetuning_type": "lora",
        "export_dir": str(MODELS_DIR / "qwen05b_gsm8k_merged"),
        "export_size": 2,
        "export_device": "cpu",
        "export_legacy_format": False
    }
    
    merge_config_path = CONFIG_DIR / MERGE_CONFIG_FILE
    
    # 转换为YAML格式
    yaml_content = "\n".join([f"{k}: {v}" for k, v in merge_config.items()])
    
    with open(merge_config_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    logger.info(f"✓ Merge config created: {merge_config_path}")
    return merge_config_path

def merge_model():
    """合并LoRA权重到基础模型"""
    logger.info("Merging LoRA weights with base model...")
    
    merge_config_path = create_merge_config()
    
    # 构建合并命令
    merge_cmd = f"llamafactory-cli export {merge_config_path}"
    
    try:
        run_command(merge_cmd, cwd=LLAMAFACTORY_DIR)
        logger.info(f"✓ Model merged successfully to: {MODELS_DIR / 'qwen05b_gsm8k_merged'}")
        return True
    except subprocess.CalledProcessError:
        logger.error("Model merging failed")
        return False

def print_summary():
    """打印训练总结"""
    logger.info("\n" + "=" * 80)
    logger.info("LLAMAFACTORY SFT TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"📁 Project directory: {PROJECT_ROOT}")
    logger.info(f"🤖 Base model: {BASE_MODEL_PATH}")
    logger.info(f"📊 Training data: {TRAIN_DATA_PATH}")
    logger.info(f"⚙️  Training config: {CONFIG_DIR / TRAIN_CONFIG_FILE}")
    logger.info(f"💾 LoRA weights: {SAVES_DIR / 'qwen05b_lora_sft'}")
    logger.info(f"🔗 Merged model: {MODELS_DIR / 'qwen05b_gsm8k_merged'}")
    logger.info(f"📈 TensorBoard logs: {LOGS_DIR}")
    logger.info("")
    logger.info("🚀 NEXT STEPS:")
    logger.info("   1. 检查训练日志和loss曲线")
    logger.info("   2. 使用WebUI测试模型效果")
    logger.info("   3. 评估模型在测试集上的性能")
    logger.info("   4. 根据效果调整超参数重新训练")
    logger.info("")
    logger.info("📋 USEFUL COMMANDS:")
    logger.info(f"   - TensorBoard: tensorboard --logdir={LOGS_DIR}")
    logger.info(f"   - WebUI: cd {LLAMAFACTORY_DIR} && llamafactory-cli webui")
    logger.info("=" * 80)

# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    global logger
    logger = setup_logging()
    
    logger.info("Starting Qwen2.5-0.5B GSM8K SFT Training with LLaMA-Factory")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    try:
        # 1. 检查前置条件
        check_prerequisites()
        
        # 2. 安装 LLaMA-Factory
        install_llamafactory()
        
        # 3. 准备配置
        prepare_llamafactory_config()
        
        # 4. 启动训练
        training_success = start_training()
        
        if training_success:
            # 5. 合并模型（可选）
            logger.info("\nDo you want to merge LoRA weights with base model? (y/n): ")
            # 自动合并，生产环境可以改为交互式
            merge_success = merge_model()
            
            if merge_success:
                logger.info("✅ Training and merging completed successfully!")
            else:
                logger.warning("⚠️ Training completed but merging failed")
        else:
            logger.error("❌ Training failed")
            return 1
        
        # 6. 打印总结
        print_summary()
        
        return 0
        
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        logger.exception("Full traceback:")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)