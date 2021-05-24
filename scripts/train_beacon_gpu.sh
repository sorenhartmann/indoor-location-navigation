 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1:mode=exclusive_process"
 #BSUB -J train_beacon
 #BSUB -n 1
 #BSUB -W 24:00
 #BSUB -B
 #BSUB -N
 #BSUB -R "rusage[mem=4GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err

module load python3/3.8.4
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8
module load ffmpeg/4.2.2

python3 src/train_model.py --n-epochs=500 beacon

