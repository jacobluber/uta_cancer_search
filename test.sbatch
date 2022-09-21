#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --error=/home/mxn2498/projects/uta_cancer_search/logs/test.err
#SBATCH --output=/home/mxn2498/projects/uta_cancer_search/logs/test.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate vae || fail "conda activate fail"
python /home/mxn2498/projects/uta_cancer_search/color/small_patches.py --batch-size 32 --epochs 30 --gpus 1 --nodes 1 --workers 10 --accelerator gpu --logging-name test_logs --patches 500 --patch-size 125 --train-size 4000 --test-size 1000 --enc-dim 2048 --latent-dim 64 --resnet resnet50 --write-coords || fail "python fail"
