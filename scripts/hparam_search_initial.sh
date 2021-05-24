 #!/bin/sh
 #BSUB -q hpc
 #BSUB -J hp-search-initial
 #BSUB -n 4
 #BSUB -W 24:00
 #BSUB -B
 #BSUB -N
 #BSUB -R span[hosts=1]
 #BSUB -R "rusage[mem=4GB]"
 #BSUB -o logs/initial_hparam_%J.out
 #BSUB -e logs/initial_hparam_%J.err

module load python3/3.8.4

python3 src/hparam_search.py --n-epochs=500 --name=initial initial

