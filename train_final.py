#!/usr/bin/env python3
"""
LLamaFactory 早停训练脚本（最终修复版）

问题说明：
    - LLaMA-Factory 的 YAML 配置不支持 early_stopping_patience 参数
    - 早停必须通过 Callback 注入实现
    - 本脚本确保 EarlyStoppingCallback 正确注入

使用方法：
    python train_with_early_stopping.py --config qwen_sft_final.yaml
    
    # 自定义早停参数
    python train_with_early_stopping.py --config qwen_sft_final.yaml --patience 3 --threshold 0.01
"""

import os
import sys
import yaml
import argparse
from pathlib import Path


def print_banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def run_training_with_early_stopping(config_path: str, patience: int = 5, threshold: float = 0.001):
    """
    使用 LLaMA-Factory Python API 运行训练，并注入 EarlyStoppingCallback
    """
    
    # 1. 转换为绝对路径
    config_path = os.path.abspath(config_path)
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    print(f"📄 配置文件: {config_path}")
    
    # 2. 读取 YAML 配置
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    
    # 移除不支持的参数（如果有的话）
    unsupported_keys = ['early_stopping_patience', 'early_stopping_threshold']
    for key in unsupported_keys:
        if key in yaml_config:
            print(f"⚠️ 移除不支持的参数: {key}")
            del yaml_config[key]
    
    print(f"📋 解析到 {len(yaml_config)} 个配置项")
    
    # 3. 创建 EarlyStoppingCallback
    from transformers.trainer_callback import EarlyStoppingCallback
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=threshold
    )
    
    print_banner("早停配置")
    print(f"  patience: {patience} epochs（连续无改善则停止）")
    print(f"  threshold: {threshold}（改善阈值）")
    print(f"  metric: eval_loss（越小越好）")
    
    # 4. 将 YAML 配置转换为命令行参数格式
    args_list = []
    for key, value in yaml_config.items():
        if value is not None and str(key).strip() != '':
            if isinstance(value, bool):
                if value:
                    args_list.append(f"--{key}")
            else:
                args_list.append(f"--{key}")
                args_list.append(str(value))
    
    # 5. 导入 LLaMA-Factory 并运行
    try:
        from llamafactory.hparams import get_train_args
        from llamafactory.train.sft import run_sft
        
        # 解析参数
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args_list)
        
        # 打印训练信息
        print_banner("训练信息")
        print(f"  模型: {model_args.model_name_or_path}")
        print(f"  数据集: {data_args.dataset}")
        print(f"  验证集比例: {data_args.val_size}")
        print(f"  Epochs: {training_args.num_train_epochs}")
        print(f"  学习率: {training_args.learning_rate}")
        print(f"  评估策略: {training_args.eval_strategy}")
        print(f"  输出目录: {training_args.output_dir}")
        
        # 检查关键配置
        if not training_args.load_best_model_at_end:
            print("\n⚠️ 警告: load_best_model_at_end=False，早停可能无法正常工作")
            print("   建议在 YAML 中设置 load_best_model_at_end: true")
        
        if training_args.eval_strategy == "no":
            print("\n❌ 错误: eval_strategy=no，早停无法工作！")
            print("   必须设置 eval_strategy: epoch 或 eval_strategy: steps")
            sys.exit(1)
        
        print_banner("开始训练")
        print("🚀 训练已启动，早停 Callback 已注入...\n")
        
        # 运行 SFT 训练，注入 callbacks
        run_sft(
            model_args, 
            data_args, 
            training_args, 
            finetuning_args, 
            generating_args, 
            callbacks=[early_stopping_callback]  # 关键：注入早停 Callback
        )
        
        print_banner("训练完成")
        print("✅ 训练成功完成！")
        print(f"📁 模型保存在: {training_args.output_dir}")
        
    except ImportError as e:
        print(f"\n❌ 导入 LLaMA-Factory 失败: {e}")
        print("\n请确保已正确安装 LLaMA-Factory:")
        print("  pip install llamafactory")
        print("  或")
        print("  cd LLaMA-Factory && pip install -e .")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="LLaMA-Factory 早停训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本使用
    python train_with_early_stopping.py --config qwen_sft_final.yaml
    
    # 自定义早停参数
    python train_with_early_stopping.py --config qwen_sft_final.yaml --patience 3
    
    # 更严格的早停阈值
    python train_with_early_stopping.py --config qwen_sft_final.yaml --patience 5 --threshold 0.01
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="YAML 配置文件路径"
    )
    
    parser.add_argument(
        "--patience", "-p",
        type=int,
        default=5,
        help="早停 patience，连续多少个 epoch 无改善则停止（默认: 5）"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.001,
        help="早停改善阈值，loss 改善小于此值视为无改善（默认: 0.001）"
    )
    
    args = parser.parse_args()
    
    print_banner("LLaMA-Factory 早停训练")
    
    run_training_with_early_stopping(
        config_path=args.config,
        patience=args.patience,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()