#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: sh run/infer.sh --exp_dir=EXP_DIR --config=CONFIG --ckpt_name=CKPT_NAME"
    exit 1
fi

for arg in "$@"; do
    case $arg in
        --exp_dir=*)
            exp_dir="${arg#*=}"
            shift
            ;;
        --config=*)
            config="${arg#*=}"
            shift
            ;;
        --ckpt_name=*)
            ckpt_name="${arg#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $arg"
            exit 1
            ;;
    esac
done

echo "Current ckpt: $ckpt_name"

export PYTHONPATH=$.:$PYTHONPATH

split_total=2

for split_idx in $(seq 0 $((split_total - 1))); do
    echo "Running split ${split_idx}/${split_total}"

    python -u run/validation.py \
        --config="${config}" \
        --split_idx="${split_idx}" \
        --split_total="${split_total}" \
        save_path "${exp_dir}" \
        resume "${exp_dir}/model/${ckpt_name}" \
        2>&1 | tee "${exp_dir}/infer-${ckpt_name}-all${split_idx}.log"
done