import torch
import os
import argparse
import time
import socket
# try:
#     import horovod.torch as hvd
# except ModuleNotFoundError:
#     pass

def add_args_gfl(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # System Setting
    parser.add_argument('--log_dir', type=str, default='./log', help='where is log file')
    parser.add_argument('--print_freq', type=int, default=1, help='frequency to print log file')
    parser.add_argument('--tag', type=str, default='GFL', metavar='T', help='task name')

    # Problem Setting
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset, default Cora, options: CiteSeer, PubMed')
    parser.add_argument('--gnn_size', type=str, default='2_1', help='neural network size used in training')
    parser.add_argument('--num_clients', type=int, default=8, help='total number of agents (default: 8)')
    parser.add_argument('--hidden_size', type=int, default=512, help='transformer dimension, devidable by num_client, (default: 512)')
    parser.add_argument('--num_classes', type=int, default=7, help='number of classes in the data')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout percentage of client graphs')
    parser.add_argument('--residual', action='store_true', help='whether use residual blocks')
    parser.add_argument('--mu', type=float, default=None, help='residual weight (default: None)')
    parser.add_argument('--FFN', action='store_true', help='use feedforward network between GCN blocks (default: True)')
    parser.add_argument('--gat', action='store_true', help='use GAT blocks (default: False)')
    
    # Algorithm Setting
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--expand_size', type=int, default=4, help='node expanding size for training (default: 4)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--global_iter', type=int, default=100, help='number of communication iterations')
    parser.add_argument('--local_iter', type=int, default=100, help='number of local iterations')
    parser.add_argument('--method', type=str, default='cat', help='aggregsation method (agg/cat) (default: cat)')
    parser.add_argument('--v1', action='store_true', help='whether use optimization algorithm v1 (default: True)')
    parser.add_argument('--q', type=int, default=4, help='number of local iteration in v2')

    args = parser.parse_args()
    args.logfile = args.log_dir+'/'+args.tag+'_D'+args.dataset+'_S'+args.gnn_size+'_C'+str(args.num_clients)+'_B'+str(args.batch_size)+'_G'+str(args.global_iter)+'_H'+str(args.hidden_size)+'_LR'+str(args.lr)+'.csv'
    if args.residual:
        args.gat = False
    with open(args.logfile,'w') as fp:
        print('dataset, model, client, batch, expand, hidden, lr, global_iter, local_iter',file= fp)
        print(args.dataset, args.gnn_size, args.num_clients, args.batch_size, args.expand_size, args.hidden_size, args.lr, args.global_iter, args.local_iter, sep=',', file = fp)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args.device)
    return args

def add_args_gfl_dist(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # System Setting
    parser.add_argument('--log_dir', type=str, default='/nobackup/users/zhan6234/GNN/log', help='where is log file')
    parser.add_argument('--print_freq', type=int, default=1, help='frequency to print log file')
    parser.add_argument('--tag', type=str, default='GFL', metavar='T', help='task name')
    parser.add_argument('--backend', type = str, default='ddl')
    parser.add_argument('--init_method', type = str, default='env://')
    parser.add_argument('--master_addr', default='127.0.0.1', type=str, help='master address used to set up distributed training')
    parser.add_argument('--master_port', default='40100', type=str, help='master port used to set up distributed training')
    

    # Problem Setting
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset, default Cora, options: CiteSeer, PubMed')
    parser.add_argument('--gnn_size', type=str, default='2_1', help='neural network size used in training')
    parser.add_argument('--num_clients', type=int, default=8, help='total number of agents (default: 8)')
    parser.add_argument('--hidden_size', type=int, default=512, help='transformer dimension, devidable by num_client, (default: 512)')
    parser.add_argument('--num_classes', type=int, default=7, help='number of classes in the data')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout percentage of client graphs')
    parser.add_argument('--residual', action='store_true', help='whether use residual blocks')
    parser.add_argument('--mu', type=float, default=None, help='residual weight (default: None)')
    parser.add_argument('--FFN', action='store_true', help='use feedforward network between GCN blocks (default: True)')
    parser.add_argument('--gat', action='store_true', help='use GAT blocks (default: False)')
    
    # Algorithm Setting
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--expand_size', type=int, default=4, help='node expanding size for training (default: 4)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--global_iter', type=int, default=100, help='number of communication iterations')
    parser.add_argument('--local_iter', type=int, default=100, help='number of local iterations')
    parser.add_argument('--method', type=str, default='cat', help='aggregsation method (agg/cat) (default: cat)')
    parser.add_argument('--v1', action='store_true', help='whether use optimization algorithm v1 (default: True)')
    parser.add_argument('--q', type=int, default=4, help='number of local iteration in v2')

    args = parser.parse_args()
    args.logfile = args.log_dir+'/'+args.tag+'_D'+args.dataset+'_S'+args.gnn_size+'_C'+str(args.num_clients)+'_B'+str(args.batch_size)+'_G'+str(args.global_iter)+'_H'+str(args.hidden_size)+'_LR'+str(args.lr)+'.csv'
    if args.residual:
        args.gat = False
    try:
        import horovod.torch as hvd
    except ModuleNotFoundError:
        pass
    hvd.init()
    args.rank = hvd.rank()
    # assert(hvd.size()>= args.num_clients+1)
    if "SLURM_JOB_NODELIST" in os.environ:
        # args.rank = int(os.environ["SLURM_PROCID"])
        print("rank:", str(args.rank))
        
        jobid = os.environ["SLURM_JOBID"]
        hostfile = "/nobackup/users/zhan6234/GNN/dist_url." + jobid  + ".txt"
        if args.rank == args.num_clients:
            print("host")
            ip = socket.gethostbyname(socket.gethostname())
            print("ip:", ip)
            port = args.master_port#find_free_port()
            args.init_method = "tcp://{}:{}".format(ip, port)
            with open(hostfile, "w+") as f:
                f.write(args.init_method)
        #     args.init_method = "tcp://{}:{}".format(ip, port)
            
        else:
            print("client")
            while not os.path.exists(hostfile):
                time.sleep(1)
            with open(hostfile, "r") as f:
                args.init_method = f.read()
        # args.rank, args.master_addr = _process_nodel_list()#int(os.environ["OMPI_COMM_WORLD_RANK"])
        #  = os.environ["MASTER_ADDR"]
        # args.master_port = #os.environ["MASTER_PORT"]
    else:
        args.rank = 0
        
    # args.rank = 
    # 
    if args.rank == args.num_clients:
        with open(args.logfile,'w') as fp:
            print('dataset, model, client, batch, expand, hidden, lr, global_iter, local_iter',file= fp)
            print(args.dataset, args.gnn_size, args.num_clients, args.batch_size, args.expand_size, args.hidden_size, args.lr, args.global_iter, args.local_iter, sep=',', file = fp)
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        assert(num_gpu >= args.num_clients),"not enough gpu!"
        args.device = torch.device('cuda:%d'%args.rank)
    else:
        args.device = torch.device('cpu')
    print(args.device)
    return args

# def _process_node_list():
#     node_list = os.environ["SLURM_JOB_NODELIST"]

def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]