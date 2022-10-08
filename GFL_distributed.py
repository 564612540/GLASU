import torch
import os
import torch.distributed as dist
from fedframe.utils.arg_parsers import add_args_gfl_dist
from fedframe.optimizers.local import GFL_dist
from fedframe.optimizers.server import GFL_server
from fedframe.aggregator.gnn_cummunicator import ServerCommunicatorTorch, ClienCommunicatorTorch
from fedframe.model.gnn import ClientGNNLinearAvg
from fedframe.dataset.loader import load_dist_graph_dataset
from fedframe.logger.logger import file_logger_centralized
import horovod.torch as hvd
import argparse

def wrapper():
    args = add_args_gfl_dist(argparse.ArgumentParser(description='GFL'))
    if args.device != 'cpu':
        torch.cuda.set_device(args.device)
    print("here, rank:", str(args.rank))
    data_train, extra_data = load_dist_graph_dataset(args.dataset, args.num_clients, args.rank, args.dropout, args.gat)
    layer_list = ClientGNNLinearAvg.get_layer_list(args.gnn_size)
    print("start running: %d"%int(args.rank))
    # device = 'cuda:%d'%args.rank
    if args.rank == args.num_clients:
        # server process
        comm = ServerCommunicatorTorch(args.num_clients, args.backend, args.init_method, args.device)
        agent = GFL_server.GFLServerAvg(data_train, extra_data, args.num_clients, args.batch_size, len(layer_list), args.hidden_size, args.device)
        log_file = file_logger_centralized(args.logfile, 4, ["global_acc","global_loss"])
        GFL_server.run(args.global_iter, args.local_iter, args.print_freq, agent, comm, log_file, args.v1)
    elif args.rank < args.num_clients:
        # client process
        if args.v1:
            agent = GFL_dist.GFLClientSv1(data_train, args.num_clients, args.rank, args.expand_size, args.device, layer_list)
        else:
            agent = GFL_dist.GFLClientSv2(data_train, args.num_clients, args.rank, args.expand_size, args.device, layer_list, args.q)
        comm = ClienCommunicatorTorch(args.num_clients, args.rank, args.backend, args.init_method, args.device)
        client_model = ClientGNNLinearAvg("C"+str(args.rank), args.gnn_size, extra_data, args.hidden_size, args.num_classes, residual= args.residual, mu = args.mu, FFN = args.FFN, GAT = args.gat)
        GFL_dist.run(args.global_iter, args.local_iter, args.print_freq, agent, comm, client_model, args)

wrapper()