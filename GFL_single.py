# from turtle import update
import torch
import os
import argparse

from fedframe.utils import arg_parsers
from fedframe.dataset.loader import load_graph_dataset
from fedframe.model.gnn import ClientGNNLinearAvg, ClientGNNLinearCat
from fedframe.optimizers.local import GFL_local
from fedframe.optimizers.server import GFL_global
from fedframe.logger.logger import file_logger_centralized
from fedframe.aggregator.gnn_cummunicator import FakeCommunicatorAvg, FakeCommunicatorCat

if __name__ == "__main__":
    args = arg_parsers.add_args_gfl(argparse.ArgumentParser(description='GFL'))
    client_datasets, client_input_size, server_train_dataset, server_test_dataset = load_graph_dataset(args.dataset, args.num_clients, args.dropout, args.gat)
    log_file = file_logger_centralized(args.logfile, 1, ["global_acc","global_loss"])
    client_datasets = [client_datasets[2]]

    if args.method == "cat":
        client_models = [ClientGNNLinearCat("GNN"+str(client_id), args.gnn_size, client_input_size[client_id], args.hidden_size, args.num_classes, 1, FFN = args.FFN) for client_id in range(1)]
        train_communicator = FakeCommunicatorCat(1, server_train_dataset, batch_size= args.batch_size)
        test_communicator = FakeCommunicatorCat(1, server_test_dataset, batch_size= args.batch_size)
        layer_list = ClientGNNLinearCat.get_layer_list(args.gnn_size)
        local_optimizer = GFL_local.GFLClientSv1(client_datasets, 1, args.expand_size, args.device, layer_list)
    elif args.method == "avg":
        client_models = [ClientGNNLinearAvg("GNN"+str(client_id), args.gnn_size, client_input_size[client_id], args.hidden_size, args.num_classes, residual= args.residual, mu = args.mu, FFN = args.FFN) for client_id in range(2,3)]
        train_communicator = FakeCommunicatorAvg(1, server_train_dataset, batch_size= args.batch_size)
        test_communicator = FakeCommunicatorAvg(1, server_test_dataset, batch_size= args.batch_size)
        layer_list = ClientGNNLinearAvg.get_layer_list(args.gnn_size)
        if args.v1:
            local_optimizer = GFL_local.GFLClientSv1(client_datasets, 1, args.expand_size, args.device, layer_list)
        else:
            local_optimizer = GFL_local.GFLClientSv2(client_datasets, 1, args.expand_size, args.device, layer_list, args.q)

    global_optimizer = GFL_global.GFL_server_serial(args.device, True)

    for global_iter in range(args.global_iter):
        if global_iter % 8 == 0:
            update = True
        else:
            update = False
        client_models = local_optimizer.train(client_models, args.local_iter, global_iter, train_communicator, args.lr, args.momentum, update)
        client_models = global_optimizer.update(client_models, global_iter)
        if global_iter % args.print_freq == 0:
            server_acc, server_loss = local_optimizer.test(client_models, global_iter, test_communicator)
            log_file.update([global_iter], [server_acc,server_loss])
            print(global_iter,server_acc)
    server_acc, server_loss = local_optimizer.test(client_models, args.global_iter, test_communicator)
    log_file.update([args.global_iter], [server_acc,server_loss])
    print(args.global_iter,server_acc)
