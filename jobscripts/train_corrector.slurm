#!/usr/bin/bash
#SBATCH --job-name Crct-Train
#SBATCH --account OPEN-30-47
#SBATCH --partition qgpu
#SBATCH --gpus 8
#SBATCH --time 12:00:00
#SBATCH --error=/mnt/proj1/open-28-36/chlebik/logdir/err_crct.log
#SBATCH --output=/mnt/proj1/open-28-36/chlebik/logdir/out_crct.log
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user=xchleb07@stud.fit.vut.cz

ml purge
ml Python/3.10.4-GCCcore-11.3.0
ml CUDA/12.4.0

cd /home/xchleb07/dev/dlbs/
source /mnt/proj1/open-28-36/chlebik/venvs/karolina/.dlbs_tf/bin/activate

DISABLE_TQDM=True
DISABLE_TRACING=True

python main.py --config "config/corrector/corrector_train_config.py" --workdir "/mnt/proj1/open-28-36/chlebik/karolina/corrector/" --dataset_file_path "/mnt/proj1/open-28-36/chlebik/datasets/1_7/128_16_(1.5, 2.8)_(1.0, 1.85)_1_6000_complex64_95bf8365c4b3dc206630fbbb8ccf3f30.npz"
