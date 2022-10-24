#!/bin/bash

#SBATCH --job-name=breast_latent64_npatches20000_batchsize
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@uta.edu
#SBATCH --partition=conference
#SBATCH --account=conference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/uta_cancer_search/logs/slurm_logs/breast_latent64_npatches20000_batchsize.err
#SBATCH --output=/home/mxn2498/projects/uta_cancer_search/logs/slurm_logs/breast_latent64_npatches20000_batchsize.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate vae || fail "conda activate fail"

cd /home/mxn2498/projects/uta_cancer_search/


python /home/mxn2498/projects/uta_cancer_search/main_gdc_umap.py \
    --everything_seed 0 \
    --test_random_seed 1 \
    --train_val_random_seed 2 \
    --pathcing_seed 3 \
    --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs \
    --logging_name breast_latent64_npatches20000_batchsize32 \
    --inv_transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_latent64_npatches20000_batchsize32 \
    --cancer_type "Breast" \
    --ratio_per_type 1 \
    --split_ratio 0.5 0.25 0.25 \
    --patch_size 64 \
    --num_patches_per_image 73 \
    --num_dataset_workers 64 \
    --batch_size 32 \
    --num_dataloader_workers 8 \
    --normalize_transform \
    --input_height 64 \
    --latent_dim 64 \
    --lr 0.0001 \
    --gradient_clip_val 0.5 \
    --num_nodes 1 \
    --strategy ddp\
    --accelerator gpu \
    --devices 1 \
    --max_epochs 100 \
    --auto_lr_find False \
    --auto_scale_batch_size False \
    --benchmark True \
    --fast_dev_run False \
    --coords_read_dir /home/mxn2498/projects/uta_cancer_search/logs/coords/breast_latent16_npatches20000_umap \
    || fail "python fail" \
    /


python /home/mxn2498/projects/uta_cancer_search/main_gdc_umap.py \
    --everything_seed 0 \
    --test_random_seed 1 \
    --train_val_random_seed 2 \
    --pathcing_seed 3 \
    --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs \
    --logging_name breast_latent64_npatches20000_batchsize64 \
    --inv_transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_latent64_npatches20000_batchsize64 \
    --cancer_type "Breast" \
    --ratio_per_type 1 \
    --split_ratio 0.5 0.25 0.25 \
    --patch_size 64 \
    --num_patches_per_image 73 \
    --num_dataset_workers 64 \
    --batch_size 64 \
    --num_dataloader_workers 8 \
    --normalize_transform \
    --input_height 64 \
    --latent_dim 64 \
    --lr 0.0001 \
    --gradient_clip_val 0.5 \
    --num_nodes 1 \
    --strategy ddp\
    --accelerator gpu \
    --devices 1 \
    --max_epochs 100 \
    --auto_lr_find False \
    --auto_scale_batch_size False \
    --benchmark True \
    --fast_dev_run False \
    --coords_read_dir /home/mxn2498/projects/uta_cancer_search/logs/coords/breast_latent16_npatches20000_umap \
    || fail "python fail" \
    /


python /home/mxn2498/projects/uta_cancer_search/main_gdc_umap.py \
    --everything_seed 0 \
    --test_random_seed 1 \
    --train_val_random_seed 2 \
    --pathcing_seed 3 \
    --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs \
    --logging_name breast_latent64_npatches20000_batchsize256 \
    --inv_transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_latent64_npatches20000_batchsize256 \
    --cancer_type "Breast" \
    --ratio_per_type 1 \
    --split_ratio 0.5 0.25 0.25 \
    --patch_size 64 \
    --num_patches_per_image 73 \
    --num_dataset_workers 64 \
    --batch_size 256 \
    --num_dataloader_workers 8 \
    --normalize_transform \
    --input_height 64 \
    --latent_dim 64 \
    --lr 0.0001 \
    --gradient_clip_val 0.5 \
    --num_nodes 1 \
    --strategy ddp\
    --accelerator gpu \
    --devices 1 \
    --max_epochs 100 \
    --auto_lr_find False \
    --auto_scale_batch_size False \
    --benchmark True \
    --fast_dev_run False \
    --coords_read_dir /home/mxn2498/projects/uta_cancer_search/logs/coords/breast_latent16_npatches20000_umap \
    || fail "python fail" \
    /
    