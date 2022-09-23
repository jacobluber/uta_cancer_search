#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --error=/home/mxn2498/projects/uta_cancer_search/logs/smallPatchesTrans_lung_64_128_10440.err
#SBATCH --output=/home/mxn2498/projects/uta_cancer_search/logs/smallPatchesTrans_lung_64_128_10440.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate vae || fail "conda activate fail"
python /home/mxn2498/projects/uta_cancer_search/color/small_patches_trans2.py --batch-size 32 --epochs 100 --gpus 1 --nodes 1 --workers 8 --accelerator gpu --cancer-type 'Bronchus and lung' --logging-name smallPatchesTrans_lung_64_128_10440 --patches 18 --patch-size 64 --train-size 9396 --test-size 1044 --enc-dim 2048 --latent-dim 128 --resnet resnet50 || fail "python fail"
