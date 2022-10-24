#!/bin/bash

#SBATCH --job-name=cifar_batch128_patch32_lr_find
#SBATCH --partition=conference
#SBATCH --account=conference
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/uta_cancer_search/logs/slurm_logs/cifar_batch128_patch32_lr_find.err
#SBATCH --output=/home/mxn2498/projects/uta_cancer_search/logs/slurm_logs/cifar_batch128_patch32_lr_find.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate vae || fail "conda activate fail"

cd /home/mxn2498/projects/uta_cancer_search/


python main_cifar.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed  2 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs --logging_name cifar_batch128_latent8_patch32_lr_find --inv_transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/cifar_batch128_latent8_patch32_lr_find --data_dir /home/data/mxn2498/cifar10/data/ --split_ratio 0.9 0.05 0.05 --no-per_image_normalize --batch_size 128 --num_dataloader_workers 8 --normalize_transform --input_height 32 --latent_dim 8 --lr 0.0001 --gradient_clip_val 0.5 --num_nodes 1 --accelerator gpu --devices 1 --max_epochs 5 --auto_lr_find True --benchmark True --fast_dev_run False || fail "python fail"
python main_cifar.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed  2 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs --logging_name cifar_batch128_latent16_patch32_lr_find --inv_transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/cifar_batch128_latent16_patch32_lr_find --data_dir /home/data/mxn2498/cifar10/data/ --split_ratio 0.9 0.05 0.05 --no-per_image_normalize --batch_size 128 --num_dataloader_workers 8 --normalize_transform --input_height 32 --latent_dim 16 --lr 0.0001 --gradient_clip_val 0.5 --num_nodes 1 --accelerator gpu --devices 1 --max_epochs 5 --auto_lr_find True --benchmark True --fast_dev_run False || fail "python fail"
# python main_cifar.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed  2 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs --logging_name cifar_batch128_latent32_patch32_lr_find --inv_transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/cifar_batch128_latent32_patch32_lr_find --data_dir /home/data/mxn2498/cifar10/data/ --split_ratio 0.9 0.05 0.05 --no-per_image_normalize --batch_size 128 --num_dataloader_workers 8 --normalize_transform --input_height 32 --latent_dim 32 --lr 0.0001 --gradient_clip_val 0.5 --num_nodes 1 --accelerator gpu --devices 1 --max_epochs 5 --auto_lr_find True --benchmark True --fast_dev_run False || fail "python fail"
python main_cifar.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed  2 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs --logging_name cifar_batch128_latent64_patch32_lr_find --inv_transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/cifar_batch128_latent64_patch32_lr_find --data_dir /home/data/mxn2498/cifar10/data/ --split_ratio 0.9 0.05 0.05 --no-per_image_normalize --batch_size 128 --num_dataloader_workers 8 --normalize_transform --input_height 32 --latent_dim 64 --lr 0.0001 --gradient_clip_val 0.5 --num_nodes 1 --accelerator gpu --devices 1 --max_epochs 5 --auto_lr_find True --benchmark True --fast_dev_run False || fail "python fail"
python main_cifar.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed  2 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs --logging_name cifar_batch128_latent128_patch32_lr_find --inv_transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/cifar_batch128_latent128_patch32_lr_find --data_dir /home/data/mxn2498/cifar10/data/ --split_ratio 0.9 0.05 0.05 --no-per_image_normalize --batch_size 128 --num_dataloader_workers 8 --normalize_transform --input_height 32 --latent_dim 128 --lr 0.0001 --gradient_clip_val 0.5 --num_nodes 1 --accelerator gpu --devices 1 --max_epochs 5 --auto_lr_find True --benchmark True --fast_dev_run False || fail "python fail"
python main_cifar.py --everything_seed 0 --test_random_seed 1 --train_val_random_seed  2 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs --logging_name cifar_batch128_latent256_patch32_lr_find --inv_transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/cifar_batch128_latent256_patch32_lr_find --data_dir /home/data/mxn2498/cifar10/data/ --split_ratio 0.9 0.05 0.05 --no-per_image_normalize --batch_size 128 --num_dataloader_workers 8 --normalize_transform --input_height 32 --latent_dim 256 --lr 0.0001 --gradient_clip_val 0.5 --num_nodes 1 --accelerator gpu --devices 1 --max_epochs 5 --auto_lr_find True --benchmark True --fast_dev_run False || fail "python fail"
