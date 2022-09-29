#!/bin/bash

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=240G
#SBATCH --gres=gpu:3
#SBATCH --time=48:00:00
#SBATCH --error=/home/mxn2498/projects/uta_cancer_search/logs/3GPU_smallPatchesTrans_lung_64_64_498800.err
#SBATCH --output=/home/mxn2498/projects/uta_cancer_search/logs/3GPU_smallPatchesTrans_lung_64_64_498800.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate vae || fail "conda activate fail"
python /home/mxn2498/projects/uta_cancer_search/color/small_patches_trans2.py --batch-size 256 --epochs 100 --nodes 1 --accelerator gpu --gpus 3 --strategy ddp --workers 8 --cancer-type 'Bronchus and lung' --logging-name 3GPU_smallPatchesTrans_lung_64_64_498800 --patches 860 --patch-size 64 --train-size 473860 --test-size 24940 --enc-dim 2048 --latent-dim 64 --resnet resnet50 --read-coords --custom-coords-file /home/mxn2498/projects/uta_cancer_search/custom_coords/smallPatchesTrans_lung_64_128_498800.data || fail "python fail"
