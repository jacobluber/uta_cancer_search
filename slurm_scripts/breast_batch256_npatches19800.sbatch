#!/bin/bash

#SBATCH --job-name=breast_batch256_npatches19800
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/uta_cancer_search/slurm_logs/breast_batch256_npatches19800.err
#SBATCH --output=/home/mxn2498/projects/uta_cancer_search/slurm_logs/breast_batch256_npatches19800.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate vae || fail "conda activate fail"

cd /home/mxn2498/projects/uta_cancer_search/

# Make sure breast_npatches19800_dataset_create.sbatch has run before this job.

python /home/mxn2498/projects/uta_cancer_search/main.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed 2 --pathcing_seed 3 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs --logging_name breast_batch256_laetent8_npathces19800 --cancer_type Breast --ratio_per_type 1 --split_ratio 0.9 0.05 0.05 --patch_size 64 --num_patches_per_image 40 --num_dataset_workers 16 --coords_read_dir /home/mxn2498/projects/uta_cancer_search/coords/breast_npathces19800 --batch_size 256 --num_dataloader_workers 8 --input_height 64 --latent_dim 8 --gradient_clip_val 0.5 --num_nodes 1 --strategy ddp --accelerator gpu --devices 2 --max_epochs 100 --auto_lr_find False --benchmark True --fast_dev_run False || fail "python fail"
python /home/mxn2498/projects/uta_cancer_search/main.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed 2 --pathcing_seed 3 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs --logging_name breast_batch256_laetent16_npathces19800 --cancer_type Breast --ratio_per_type 1 --split_ratio 0.9 0.05 0.05 --patch_size 64 --num_patches_per_image 40 --num_dataset_workers 16 --coords_read_dir /home/mxn2498/projects/uta_cancer_search/coords/breast_npathces19800 --batch_size 256 --num_dataloader_workers 8 --input_height 64 --latent_dim 16 --gradient_clip_val 0.5 --num_nodes 1 --strategy ddp --accelerator gpu --devices 2 --max_epochs 100 --auto_lr_find False --benchmark True --fast_dev_run False || fail "python fail"
python /home/mxn2498/projects/uta_cancer_search/main.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed 2 --pathcing_seed 3 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs --logging_name breast_batch256_laetent32_npathces19800 --cancer_type Breast --ratio_per_type 1 --split_ratio 0.9 0.05 0.05 --patch_size 64 --num_patches_per_image 40 --num_dataset_workers 16 --coords_read_dir /home/mxn2498/projects/uta_cancer_search/coords/breast_npathces19800 --batch_size 256 --num_dataloader_workers 8 --input_height 64 --latent_dim 32 --gradient_clip_val 0.5 --num_nodes 1 --strategy ddp --accelerator gpu --devices 2 --max_epochs 100 --auto_lr_find False --benchmark True --fast_dev_run False || fail "python fail"
python /home/mxn2498/projects/uta_cancer_search/main.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed 2 --pathcing_seed 3 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs --logging_name breast_batch256_laetent64_npathces19800 --cancer_type Breast --ratio_per_type 1 --split_ratio 0.9 0.05 0.05 --patch_size 64 --num_patches_per_image 40 --num_dataset_workers 16 --coords_read_dir /home/mxn2498/projects/uta_cancer_search/coords/breast_npathces19800 --batch_size 256 --num_dataloader_workers 8 --input_height 64 --latent_dim 64 --gradient_clip_val 0.5 --num_nodes 1 --strategy ddp --accelerator gpu --devices 2 --max_epochs 100 --auto_lr_find False --benchmark True --fast_dev_run False || fail "python fail"
python /home/mxn2498/projects/uta_cancer_search/main.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed 2 --pathcing_seed 3 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs --logging_name breast_batch256_laetent128_npathces19800 --cancer_type Breast --ratio_per_type 1 --split_ratio 0.9 0.05 0.05 --patch_size 64 --num_patches_per_image 40 --num_dataset_workers 16 --coords_read_dir /home/mxn2498/projects/uta_cancer_search/coords/breast_npathces19800 --batch_size 256 --num_dataloader_workers 8 --input_height 64 --latent_dim 128 --gradient_clip_val 0.5 --num_nodes 1 --strategy ddp --accelerator gpu --devices 2 --max_epochs 100 --auto_lr_find False --benchmark True --fast_dev_run False || fail "python fail"
python /home/mxn2498/projects/uta_cancer_search/main.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed 2 --pathcing_seed 3 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs --logging_name breast_batch256_laetent256_npathces19800 --cancer_type Breast --ratio_per_type 1 --split_ratio 0.9 0.05 0.05 --patch_size 64 --num_patches_per_image 40 --num_dataset_workers 16 --coords_read_dir /home/mxn2498/projects/uta_cancer_search/coords/breast_npathces19800 --batch_size 256 --num_dataloader_workers 8 --input_height 64 --latent_dim 256 --gradient_clip_val 0.5 --num_nodes 1 --strategy ddp --accelerator gpu --devices 2 --max_epochs 100 --auto_lr_find False --benchmark True --fast_dev_run False || fail "python fail"