#!/bin/sh
set -x 

while [ "$#" -gt 0 ]; do
    case "$1" in
        --exp_dir=*)
            exp_dir="${1#*=}"
            ;;
        --config=*)
            config="${1#*=}"
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

if [ -z "$exp_dir" ] || [ -z "$config" ]; then
    echo "Usage: sh run/train.sh --exp_dir=XX --config=XX"
    exit 1
fi

mkdir -p "${exp_dir}"

export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=3 python -u xdecoder_test/train.py \
  --config="${config}" \
  save_path "${exp_dir}" \
  2>&1 | tee -a "${exp_dir}/train-$(date +"%Y%m%d_%H%M").log"
