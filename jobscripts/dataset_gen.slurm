#!/usr/bin/bash
#SBATCH --job-name LBS-Train
#SBATCH --account OPEN-30-47
#SBATCH --partition qgpu
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --time 01:00:00
#SBATCH --error=/mnt/proj1/open-28-36/chlebik/logdir/err.log
#SBATCH --output=/mnt/proj1/open-28-36/chlebik/logdir/out.log
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user=xchleb07@stud.fit.vut.cz

ml purge
ml Python/3.10.4-GCCcore-11.3.0
ml CUDA/12.4.0

cd /home/xchleb07/dev/dlbs/
source /mnt/proj1/open-28-36/chlebik/venvs/karolina/.dlbs_tf/bin/activate

DISABLE_TQDM=True
DISABLE_TRACING=True

python dataset_generator/dataset_generator.py --config "config/dataset_gen/dataset_generation_config.py" --workdir "/mnt/proj1/open-28-36/chlebik/datasets/1_7/"
