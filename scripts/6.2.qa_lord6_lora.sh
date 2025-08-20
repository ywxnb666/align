#!/bin/bash
######################################################################
#6.2.QA_LORD6_LORA ---

# TRAINING LORD6 on QA TASKS with LORA METHODS.

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2024, ZiLiang, all rights reserved.
# Created:  4 May 2024
######################################################################

export CUDA_HOME=/mnt/petrelfs/share/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=121
export HF_ENDPOINT="https://hf-mirror.com"

# echo $PATH
# echo $LD_LIBRARY_PATH

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align2/bin/python3
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                                        --format=csv,noheader,nounits | \
                              awk -F ', ' '$2 < 100 && $3 == 0 {print $1}' | \
                              paste -sd ",")

# 如果没有找到空闲 GPU，默认使用 GPU 0
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "No idle GPUs found, using GPU 0"
else
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

export TORCH_USE_CUDA_DSA="1"
export root_dir="/mnt/petrelfs/songxinhao/sxh/workspace3/align/"
export POD_save_dir="${root_dir}qa_ckpts/"
export from_path="/mnt/petrelfs/share_data/ai4good_shared/models/meta-llama/llama3-8b-instruct"
# export from_path="google/gemma-7b"
# export TRAIN_NUMS=(64 128 256 512)
# export TRAIN_NUMS=(128 256 512)
export TRAIN_NUMS=(64)
export train_times=(1 2 3 4 5)
# export train_times=(1 2)
export msl=256
export task_ls=("truthful_qa" "allenai/ai2_arc")
# export task_ls=("truthful_qa" "allenai/ai2_arc")
# export task_ls=("piqa")
# export train_taskls=("kd")
export train_taskls=("LoRD-VI")

export is_black_box=1
export use_lora=1

# export epoch=3
# export period=3

export epoch=2
export period=1

export sub_set_num=1
export sub_stage_num=512
# export sub_stage_num=16
export max_new_tokens=32
export infer_batch_size=1
export batch_size=1

export beta=1.0
export temperature=2

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

export tau1=0.80
# export tau1=-0.1
export tau2=0.85

for train_num in ${TRAIN_NUMS[*]}
do
    for train_time in ${train_times[*]}
    do
	for task in ${task_ls[*]}
	do
	    for train_task in ${train_taskls[*]}
	    do
		echo "====================================================="
		echo "+++++++train_num: ${train_num}+++++++"
		echo "+++++++train_time: ${train_time}+++++++"
		echo "+++++++task: ${task}+++++++"
		echo "+++++++train_task: ${train_task}+++++++"
		echo "====================================================="

		export save_path="${POD_save_dir}QAAAnew${task}${train_num}${train_time}${train_task}"

		$python ${root_dir}lord_train.py\
		    --dataset_task=$task \
		    --use_lora=$use_lora \
		    --from_path=$from_path \
		    --is_black_box=$is_black_box \
		    --sub_set_num=$sub_set_num \
		    --sub_stage_num=$sub_stage_num\
		    --infer_batch_size=$infer_batch_size\
		    --tau1=$tau1 \
		    --tau2=$tau2 \
		    --task=$train_task \
		    --device="cuda" \
		    --epoch=$epoch \
		    --period_num=$period \
		    --acc_step=1 \
		    --log_step=50 \
		    --train_num=$train_num \
		    --max_new_tokens=$max_new_tokens \
		    --LR="3e-5" \
		    --beta=$beta \
		    --temperature=$temperature \
		    --batch_size=$batch_size \
		    --use_old_logits=$use_old_logits\
		    --use_vic_logits=$use_vic_logits\
		    --use_kld=$use_kld\
		    --max_length=$msl \
		    --save_path=$save_path
		echo "DONE FOR ONE TRAIN NUMBERS...."
	    done
	done
    done
done

echo "NOW BEGIN TO INFERENCE..."
$python ${root_dir}qa_process.py


echo "RUNNING 6.2.qa_lord6_lora.sh DONE."
# 6.2.qa_lord6_lora.sh ends here
