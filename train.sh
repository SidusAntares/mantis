#!/usr/bin/env bash
set -euo pipefail  # 启用严格模式：出错即停
CONDA_BASE=$(conda info --base)
PYTHON_EXEC="$CONDA_BASE/envs/mantis/bin/python"
# ================== 配置区（集中管理，便于修改）==================
SOURCES=("france/31TCJ/2017" "france/30TXT/2017" "denmark/32VNH/2017" "austria/33UVP/2017")
SEEDS=(111)
TYPE=('head' 'full')

# ================================================================


# 执行单个训练任务的函数
run_experiment() {
    local source_path="$1"
    local type="$2"
    local seed="$3"
    echo "--------------------------------------------------"
    echo "[INFO] 开始训练: source=$source_path, fine_tuning_type $type, seed=$seed"
    echo "[CMD] "$PYTHON_EXEC" train.py --source '$source_path' --fine_tuning_type '$type'"
    echo "--------------------------------------------------"

    # 执行命令，失败则退出
    "$PYTHON_EXEC" train.py --source "$source_path" --fine_tuning_type "$type" --seed "$seed"
}

# 主循环：遍历所有组合
for source in "${SOURCES[@]}"; do
    for type in "${TYPE[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "$source" "$type" "$seed"
            run_experiment "$source" "$type" "$seed" --doy_p
        done
    done
done

echo "[SUCCESS] 所有实验已完成！共 $((${#SOURCES[@]} * ${#TYPE[@]})) 个任务。"