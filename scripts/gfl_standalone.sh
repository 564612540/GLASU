#!/bin/bash

# python3 ./GFL_standalone.py --log_dir "./log" --tag "DistV2" --num_clients 3 --hidden_size 256 --dropout 0.2 --global_iter 48 --local_iter 64 --lr 0.075 --gnn_size 2_2 --expand_size 3 --method avg --print_freq 1 --residual --batch_size 32 --mu 0.4 --momentum 0.0 --num_classes 41 --dataset Reddit

# python3 ./GFL_standalone.py --log_dir "./log" --tag "DistV1" --num_clients 3 --hidden_size 256 --dropout 0.2 --global_iter 48 --local_iter 128 --lr 0.01 --gnn_size 2_2 --expand_size 3 --method avg --print_freq 1 --residual --batch_size 32 --mu 0.4 --momentum 0.0 --num_classes 41 --dataset Reddit --v1

# python3 ./GFL_standalone.py --log_dir "./log" --tag "Cent" --num_clients 1 --hidden_size 256 --dropout 0.0 --global_iter 48 --local_iter 128 --lr 0.02 --gnn_size 2_2 --expand_size 7 --method avg --print_freq 1 --residual --batch_size 32 --mu 0.4 --momentum 0.0 --num_classes 41 --dataset Reddit --v1

python3 ./GFL_single.py --log_dir "./log" --tag "StAl" --num_clients 3 --hidden_size 256 --dropout 0.5 --global_iter 48 --local_iter 64 --lr 0.01 --gnn_size 2_2 --expand_size 2 --method avg --print_freq 1 --residual --batch_size 32 --mu 0.4 --momentum 0.0 --num_classes 41 --dataset Reddit --v1

# python3 ./GFL_standalone.py --log_dir "./log" --tag "DistV1" --num_clients 3 --hidden_size 192 --dropout 0.2 --global_iter 32 --local_iter 32 --lr 0.01 --gnn_size 2_2 --expand_size 2 --method avg --print_freq 1 --residual True --FFN False --batch_size 16 --mu 0.5 --momentum 0.1 --v1 True --num_classes 7

# python3 ./GFL_standalone.py --log_dir "./log" --tag "Cent" --num_clients 1 --hidden_size 192 --dropout 0.0 --global_iter 32 --local_iter 32 --lr 0.01 --gnn_size 1_4 --expand_size 5 --method avg --print_freq 1 --residual True --FFN False --batch_size 16 --mu 0.5 --momentum 0.1 --v1 True --num_classes 7

# python3 ./GFL_single.py --log_dir "./log" --tag "StAl" --num_clients 3 --hidden_size 192 --dropout 0.2 --global_iter 32 --local_iter 32 --lr 0.01 --gnn_size 1_4 --expand_size 2 --method avg --print_freq 1 --residual True --FFN False --batch_size 16 --mu 0.5 --momentum 0.1 --v1 True --num_classes 7