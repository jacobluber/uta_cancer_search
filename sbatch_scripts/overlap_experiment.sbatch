#!/bin/bash

#SBATCH --job-name=overlap_experiment
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=70
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --error=/home/mxn2498/projects/uta_cancer_search/logs/overlap_experiment.err
#SBATCH --output=/home/mxn2498/projects/uta_cancer_search/logs/overlap_experiment.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate vae || fail "conda activate fail"

python /home/mxn2498/projects/uta_cancer_search/color/small_patches_trans2.py --batch-size 32 --epochs 100 --gpus 1 --nodes 1 --workers 32 --accelerator gpu --cancer-type 'Bronchus and lung' --logging-name overlap_experiment_batch32_latent16 --patches 18 --patch-size 64 --train-size 9396 --test-size 1044 --enc-dim 2048 --latent-dim 16 --resnet resnet50 --read-coords --custom-coords-file /home/mxn2498/projects/uta_cancer_search/custom_coords/overlap.data || fail "python fail"
python /home/mxn2498/projects/uta_cancer_search/color/small_patches_trans2.py --batch-size 128 --epochs 100 --gpus 1 --nodes 1 --workers 32 --accelerator gpu --cancer-type 'Bronchus and lung' --logging-name overlap_experiment_batch128_latent16 --patches 18 --patch-size 64 --train-size 9396 --test-size 1044 --enc-dim 2048 --latent-dim 16 --resnet resnet50 --read-coords --custom-coords-file /home/mxn2498/projects/uta_cancer_search/custom_coords/overlap.data || fail "python fail"
python /home/mxn2498/projects/uta_cancer_search/color/small_patches_trans2.py --batch-size 32 --epochs 100 --gpus 1 --nodes 1 --workers 32 --accelerator gpu --cancer-type 'Bronchus and lung' --logging-name overlap_experiment_batch32_latent32 --patches 18 --patch-size 64 --train-size 9396 --test-size 1044 --enc-dim 2048 --latent-dim 32 --resnet resnet50 --read-coords --custom-coords-file /home/mxn2498/projects/uta_cancer_search/custom_coords/overlap.data || fail "python fail"
python /home/mxn2498/projects/uta_cancer_search/color/small_patches_trans2.py --batch-size 128 --epochs 100 --gpus 1 --nodes 1 --workers 32 --accelerator gpu --cancer-type 'Bronchus and lung' --logging-name overlap_experiment_batch128_latent32 --patches 18 --patch-size 64 --train-size 9396 --test-size 1044 --enc-dim 2048 --latent-dim 32 --resnet resnet50 --read-coords --custom-coords-file /home/mxn2498/projects/uta_cancer_search/custom_coords/overlap.data || fail "python fail"
python /home/mxn2498/projects/uta_cancer_search/color/small_patches_trans2.py --batch-size 32 --epochs 100 --gpus 1 --nodes 1 --workers 32 --accelerator gpu --cancer-type 'Bronchus and lung' --logging-name overlap_experiment_batch32_latent64 --patches 18 --patch-size 64 --train-size 9396 --test-size 1044 --enc-dim 2048 --latent-dim 64 --resnet resnet50 --read-coords --custom-coords-file /home/mxn2498/projects/uta_cancer_search/custom_coords/overlap.data || fail "python fail"
python /home/mxn2498/projects/uta_cancer_search/color/small_patches_trans2.py --batch-size 128 --epochs 100 --gpus 1 --nodes 1 --workers 32 --accelerator gpu --cancer-type 'Bronchus and lung' --logging-name overlap_experiment_batch128_latent64 --patches 18 --patch-size 64 --train-size 9396 --test-size 1044 --enc-dim 2048 --latent-dim 64 --resnet resnet50 --read-coords --custom-coords-file /home/mxn2498/projects/uta_cancer_search/custom_coords/overlap.data || fail "python fail"