 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J initial_gpu
 #BSUB -n 1
 #BSUB -W 02:00
 #BSUB -B
 #BSUB -N
 #BSUB -R "rusage[mem=4GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err

#module load python3/3.8.3
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8
module load ffmpeg/4.2.2

source /zhome/17/6/118138/indoor-location-navigation/.venv/bin/activate

python /zhome/17/6/118138/indoor-location-navigation/src/models/initial_model.py
