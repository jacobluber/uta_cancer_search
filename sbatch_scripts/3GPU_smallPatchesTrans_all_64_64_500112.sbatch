#!/bin/bash

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=240G
#SBATCH --gres=gpu:3
#SBATCH --time=48:00:00
#SBATCH --error=/home/mxn2498/projects/uta_cancer_search/logs/3GPU_smallPatchesTrans_all_64_64_500112.err
#SBATCH --output=/home/mxn2498/projects/uta_cancer_search/logs/3GPU_smallPatchesTrans_all_64_64_500112.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate vae || fail "conda activate fail"
python /home/mxn2498/projects/uta_cancer_search/color/small_patches_trans2.py --batch-size 256 --epochs 100 --gpus 3 --nodes 1 --workers 8 --accelerator gpu --cancer-type 'all' --logging-name 3GPU_smallPatchesTrans_all_64_64_500112 --patches 828 --patch-size 64 --train-size 475108 --test-size 25004 --enc-dim 2048 --latent-dim 64 --resnet resnet50 --write-coords --coords-file-name all_64_500112_10p.data || fail "python fail"
