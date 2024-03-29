#!/bin/bash

#SBATCH --job-name=breast_patchsize64_npatches99000_dataset_create
#SBATCH --partition=conference
#SBATCH --account=conference
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/uta_cancer_search/logs/slurm_logs/breast_patchsize64_npatches99000_dataset_create.err
#SBATCH --output=/home/mxn2498/projects/uta_cancer_search/logs/slurm_logs/breast_patchsize64_npatches99000_dataset_create.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate vae || fail "conda activate fail"

cd /home/mxn2498/projects/uta_cancer_search/

# This job will create ./coords/breast_npatches99000/ dataset.
python /home/mxn2498/projects/uta_cancer_search/main_gdc.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed 2 --pathcing_seed 3 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs --logging_name breast_patchsize64_npathces99000 --cancer_type Breast --ratio_per_type 1 --split_ratio 0.9 0.05 0.05 --patch_size 64 --num_patches_per_image 200 --num_dataset_workers 64 --batch_size 256 --num_dataloader_workers 8 --normalize_transform --input_height 64 --latent_dim 8 --lr 0.0001 --inv_transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_patchsize64_npatches99000_dataset_create --gradient_clip_val 0.5 --num_nodes 1 --strategy ddp --accelerator gpu --devices 1 --max_epochs 1 --auto_lr_find False --benchmark True --fast_dev_run False || fail "python fail"

# Deleting the unnecessary logs
rm -rf logs/tb_logs/breast_patchsize64_npathces99000/version_0