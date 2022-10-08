#!/bin/bash
#SBATCH -e gfl%j.err
#SBATCH -o gfl%j.out

#SBATCH --mail-user=zhan6234@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=1T
#SBATCH --exclusive
#SBATCH --time 02:00:00

module purge all
module add cuda/10.1

HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=fgl
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

export EGO_TOP=/opt/ibm/spectrumcomputing
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

export NODELIST=nodelist.txt
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

echo " Running on multiple nodes and GPU devices"
echo ""
echo " Run started at:- "
date

horovodrun -np $SLURM_NTASKS -H `cat $NODELIST` python $HOME2/GNN/GFL_distributed.py --backend gloo --log_dir "./log" --tag "Cent" --num_clients 3 --hidden_size 192 --dropout 0.0 --global_iter 32 --local_iter 64 --lr 0.005 --gnn_size 4_1 --expand_size 2 --method avg --print_freq 1 --residual --batch_size 16 --mu 0.5 --momentum 0.0  --dataset Cora --num_classes 7 --v1