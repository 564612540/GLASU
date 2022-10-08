#!/bin/bash

python3 ./GFL_standalone.py --log_dir "./log" --tag "DistV2" --num_clients 3 --hidden_size 256 --dropout 0.2 --global_iter 32 --local_iter 4 --lr 0.1 --gnn_size 2_2 --expand_size 3 --method avg --print_freq 1 --residual --batch_size 16 --mu 0.2 --momentum 0.0 --dataset Amsterdam --num_classes 9

python3 ./GFL_standalone.py --log_dir "./log" --tag "DistV1" --num_clients 3 --hidden_size 256 --dropout 0.2 --global_iter 32 --local_iter 16 --lr 0.1 --gnn_size 2_2 --expand_size 3 --method avg --print_freq 1 --residual --batch_size 16 --mu 0.2 --momentum 0.0 --dataset Amsterdam --num_classes 9 --v1

python3 ./GFL_standalone.py --log_dir "./log" --tag "Cent" --num_clients 1 --hidden_size 256 --dropout 0.2 --global_iter 32 --local_iter 16 --lr 0.1 --gnn_size 1_4 --expand_size 8 --method avg --print_freq 1 --residual --batch_size 16 --mu 0.2 --momentum 0.0 --dataset Amsterdam --num_classes 9 --v1

python3 ./GFL_single.py --log_dir "./log" --tag "StAl" --num_clients 3 --hidden_size 256 --dropout 0.2 --global_iter 32 --local_iter 16 --lr 0.1 --gnn_size 1_4 --expand_size 3 --method avg --print_freq 1 --residual --batch_size 16 --mu 0.2 --momentum 0.0 --dataset Amsterdam --num_classes 9 --v1