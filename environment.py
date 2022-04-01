"""
@TranNhiem 2022 
Script: environment.py
Configures and documents a distributed of your training environment.
+ MPI, a high-performance message passing library
+ (NCCL), NVIDIA Collective Communication Library 
MPI, NCCL work together to enable and manage fast communication between nodes in the cluster.

"""

import os
import torch
import pytorch_lightning as pl

#Testing the Machine Cuda and ML framework Envs
def print_dl_library_versions():
    print(f"CUDNN version: {torch.backends.cudnn.version()}")
    print(f"Torch version: {torch.__version__}")
    print(f"PyTorch Lightning version: {pl.__version__}")

#Testing the Multi-Node Environments
def configure_multi_node_environment(nodes):
    os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
    os.environ["MASTER_PORT"] = "6105"
    os.environ["NODE_RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    print(f"Configuring environment for {nodes} nodes.")
    print("MASTER_ADDR = {}".format(os.environ["MASTER_ADDR"]))
    print("MASTER_PORT = {}".format(os.environ["MASTER_PORT"]))
    print("NODE_RANK = {}".format(os.environ["NODE_RANK"]))