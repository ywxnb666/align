#!/bin/bash
######################################################################
# 7.0.VQ_LORD_TRAIN.SH ---
#
# VQ-LoRD 训练脚本
# 用于窃取多模态模型的图像识别能力
#
# 训练流程:
# 1. 阶段1: VQ Codebook 预训练
# 2. 阶段2: 视觉能力蒸馏 
# 3. 阶段3: LoRD 联合训练
#
# Author: VQ-LoRD Project
# Created: January 2026
######################################################################

# CUDA 环境配置
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=121
export HF_ENDPOINT="https://hf-mirror.com"

echo "HOME: ${HOME}"
export python=${HOME}/autodl-tmp/conda/envs/align_vq/bin/python3

# 自动选择空闲 GPU
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                                        --format=csv,noheader,nounits | \
                              awk -F ', ' '$2 < 100 && $3 == 0 {print $1}' | \
                              paste -sd ",")

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "未找到空闲 GPU，使用 GPU 0"
else
    echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
fi

export PYTHONIOENCODING=utf-8
export TORCH_USE_CUDA_DSA="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 项目路径
export root_dir="/root/workspace/align/"
export save_dir="${root_dir}vq_lord_ckpts/"
export data_dir="${root_dir}vq_lord_data/"

# 模型路径
export model_path="/root/autodl-tmp/models/llama3-llava-next-8b-hf"
export victim_model="gpt-4-vision-preview"

# ================== VQ-LoRD 参数配置 ==================

# VQ 参数
export vq_codebook_size=8192    # Codebook 大小
export vq_commitment_cost=0.25  # Commitment loss 权重
export freeze_vision_tower=0    # 是否冻结原始视觉编码器 (0=不冻结, 1=冻结)

# 损失权重
export alpha=1.0                # 视觉蒸馏损失权重
export beta=0.25                # VQ 损失权重
export temperature=1.5          # 蒸馏温度

# 训练参数 (针对 A800 80GB PCIe 优化)
export stage=3                  # 训练阶段 (1=VQ预训练, 2=视觉蒸馏, 3=全部)
export epochs=3                 # 训练轮数
export batch_size=1             # 批次大小 (LLaVA-Next 不支持 bs>1, 不同图片 patch 数不同)
export grad_accum=8             # 梯度累积步数 (等效batch_size=8)
export lr=2e-5                  # 学习率 (等效bs更大,lr适当降低)
export max_length=1024          # 最大序列长度 (80GB充分利用长上下文)
export max_new_tokens=256       # LoRD生成最大token数 (更完整的回复用于对比)

# LoRA 参数
export use_lora=1               # 使用 LoRA
export lora_rank=64             # LoRA rank (80GB可支持更大rank,提升拟合能力)
export lora_alpha_val=128       # LoRA alpha (通常为rank的2倍)

# 量化
export use_4bit=1               # 使用 4-bit 量化

# 数据
export train_num=500           # 训练样本数 (更多数据充分利用算力)
export dataset_name="scienceqa" # 使用 ScienceQA 数据集
export scienceqa_split="train"  # ScienceQA split
export scienceqa_eval_split="validation"  # ScienceQA 验证/测试 split
export scienceqa_seed=20240306  # ScienceQA 划分随机种子

# 日志和保存
export log_step=10
export save_step=100

# ================== 训练执行 ==================

echo "======================================================"
echo "VQ-LoRD 训练开始"
echo "======================================================"
echo "模型路径: $model_path"
echo "教师模型: $victim_model"
echo "VQ Codebook 大小: $vq_codebook_size"
echo "训练阶段: $stage"
echo "保存路径: $save_dir"
echo "======================================================"

# 创建必要目录
mkdir -p $save_dir
mkdir -p $data_dir

# 运行训练
$python ${root_dir}vq_lord/train_vq_lord.py \
    --model_path=$model_path \
    --victim_model=$victim_model \
    --vq_codebook_size=$vq_codebook_size \
    --vq_commitment_cost=$vq_commitment_cost \
    --freeze_vision_tower=$freeze_vision_tower \
    --alpha=$alpha \
    --beta=$beta \
    --temperature=$temperature \
    --stage=$stage \
    --epochs=$epochs \
    --batch_size=$batch_size \
    --lr=$lr \
    --max_length=$max_length \
    --use_lora=$use_lora \
    --lora_rank=$lora_rank \
    --lora_alpha=$lora_alpha_val \
    --use_4bit=$use_4bit \
    --grad_accum=$grad_accum \
    --max_new_tokens=$max_new_tokens \
    --data_dir=$data_dir \
    --train_num=$train_num \
    --dataset_name=$dataset_name \
    --scienceqa_split=$scienceqa_split \
    --scienceqa_eval_split=$scienceqa_eval_split \
    --scienceqa_seed=$scienceqa_seed \
    --reuse_vq_codebook=1 \
    --reuse_stage2=1 \
    --stage2_ckpt_path=$save_dir/stage2_vision \
    --save_path=$save_dir \
    --log_step=$log_step \
    --save_step=$save_step \
    --device="cuda"

echo "======================================================"
echo "VQ-LoRD 训练完成"
echo "======================================================"

# ================== 训练结果验证 ==================
# if [ "$stage" -ge 3 ]; then
#     echo "======================================================"
#     echo "开始 ScienceQA 验证"
#     echo "======================================================"

#     $python ${root_dir}sciqa_process.py \
#         --model_path=$model_path \
#         --adapter_path=$save_dir/stage3_lord_final \
#         --split=$scienceqa_eval_split \
#         --max_samples=200 \
#         --max_new_tokens=64 \
#         --use_4bit=$use_4bit \
#         --save_path=$save_dir/sciqa_eval.json

#     echo "======================================================"
#     echo "ScienceQA 验证完成"
#     echo "结果保存在: $save_dir/sciqa_eval.json"
#     echo "======================================================"
# fi

# 7.0.vq_lord_train.sh ends here
