#!/bin/bash

#SBATCH --job-name=5type_pred
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=conference
#SBATCH --account=conference
#SBATCH --mail-user=mohammadsadegh.nasr@uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/uta_cancer_search/logs/slurm_logs/5type_pred.err
#SBATCH --output=/home/mxn2498/projects/uta_cancer_search/logs/slurm_logs/5type_pred.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate vae || fail "conda activate fail"

cd /home/mxn2498/projects/uta_cancer_search/

python main_gdc_predict.py --model_checkpoint_path /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_batch128_latent64_npatches19800_resnet50/version_0/checkpoints/epoch=99-step=7800.ckpt --everything_seed 0 --pathcing_seed 1 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs --logging_name pred_lung --cancer_type ",Bronchus and lung" --ratio_per_type 1 --transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_batch128_latent64_npatches19800_resnet50 --patch_size 64 --num_patches_per_image 17 --num_dataset_workers 64 --no-per_image_normalize --batch_size 256 --num_dataloader_workers 8 --num_nodes 1 --accelerator gpu --devices 1 || fail "python fail"
python main_gdc_predict.py --model_checkpoint_path /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_batch128_latent64_npatches19800_resnet50/version_0/checkpoints/epoch=99-step=7800.ckpt --everything_seed 0 --pathcing_seed 1 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs --logging_name pred_colon --cancer_type ",Colon" --ratio_per_type 1 --transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_batch128_latent64_npatches19800_resnet50 --patch_size 64 --num_patches_per_image 38 --num_dataset_workers 64 --no-per_image_normalize --batch_size 256 --num_dataloader_workers 8 --num_nodes 1 --accelerator gpu --devices 1 || fail "python fail"
python main_gdc_predict.py --model_checkpoint_path /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_batch128_latent64_npatches19800_resnet50/version_0/checkpoints/epoch=99-step=7800.ckpt --everything_seed 0 --pathcing_seed 1 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs --logging_name pred_pancreas --cancer_type ",Pancreas" --ratio_per_type 1 --transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_batch128_latent64_npatches19800_resnet50 --patch_size 64 --num_patches_per_image 120 --num_dataset_workers 64 --no-per_image_normalize --batch_size 256 --num_dataloader_workers 8 --num_nodes 1 --accelerator gpu --devices 1 || fail "python fail"
python main_gdc_predict.py --model_checkpoint_path /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_batch128_latent64_npatches19800_resnet50/version_0/checkpoints/epoch=99-step=7800.ckpt --everything_seed 0 --pathcing_seed 1 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs --logging_name pred_prostate --cancer_type ",Prostate gland" --ratio_per_type 1 --transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_batch128_latent64_npatches19800_resnet50 --patch_size 64 --num_patches_per_image 47 --num_dataset_workers 64 --no-per_image_normalize --batch_size 256 --num_dataloader_workers 8 --num_nodes 1 --accelerator gpu --devices 1 || fail "python fail"
python main_gdc_predict.py --model_checkpoint_path /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_batch128_latent64_npatches19800_resnet50/version_0/checkpoints/epoch=99-step=7800.ckpt --everything_seed 0 --pathcing_seed 1 --logging_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs --logging_name pred_brain --cancer_type ",Brain" --ratio_per_type 1 --transformations_read_dir /home/mxn2498/projects/uta_cancer_search/logs/tb_logs/breast_batch128_latent64_npatches19800_resnet50 --patch_size 64 --num_patches_per_image 16 --num_dataset_workers 64 --no-per_image_normalize --batch_size 256 --num_dataloader_workers 8 --num_nodes 1 --accelerator gpu --devices 1 || fail "python fail"
