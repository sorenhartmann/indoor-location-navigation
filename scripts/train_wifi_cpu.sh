 #!/bin/sh
 #BSUB -q hpc
 #BSUB -J train_wifi
 #BSUB -n 16
 #BSUB -W 8:00
 #BSUB -B
 #BSUB -N
 #BSUB -R span[hosts=1]
 #BSUB -R "rusage[mem=4GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err

module load python3/3.8.4

python3 src/train_model.py --n-epochs=500 wifi wifi-model-cpu

