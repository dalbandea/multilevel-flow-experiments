from __future__ import annotations
import argparse

#############
#  PARSING  #
#############

parser = argparse.ArgumentParser()
# python3 main/flow_hmc-check.py -n NTRAJ -t TAU -ns NSTEPS
# python3 main/train-network.py -L 6 -b 0.537 -l 0.5 -B 500 -E 1000 -s 100 --wdir=results/trash/

parser.add_argument("-L", "--lsize", help="Lattice size", type=int, required=True)
parser.add_argument("-b", "--beta", help="Beta", type=float, required=True)
parser.add_argument("-l", "--lam", help="Lambda value", type=float, required=True)

parser.add_argument("-k", "--ksize", help="Kernel size of affine layer", type=int, default=3)

parser.add_argument("-B", "--batch", help="Batch size", type=int, required=True)
parser.add_argument("-E", "--nepochs", help="Number of epochs used for training", required=True, type=int)

parser.add_argument("-s", "--save", help="Save every s epochs", type=int, required=False, default=1)
parser.add_argument("-w", "--wdir", help="Working directory", type=str, default="results/trained_networks/")

parser.add_argument("--seed", help="Set torch seed", type=int, required=False)

args = parser.parse_args()

if args.ksize and args.ksize % 2 == 0:
    parser.error("Kernel size must be odd")


###################################
#  LOAD SEED FOR REPRODUCIBILITY  #
###################################

import torch
if args.seed != None:
    torch.manual_seed(args.seed)


#############
# Libraries #
#############

import sys
import os
import shutil
from datetime import datetime

# sys.path.append("../../")
sys.path.append(".")

import math
import logging

import matplotlib.pyplot as plt

import torch.nn.functional as F
import pytorch_lightning as pl

from jsonargparse.typing import PositiveInt, PositiveFloat, NonNegativeFloat

import flows.phi_four as phi_four
import flows.transforms as transforms
import flows.utils as utils
from flows.flow_hmc import *
from flows.models import MultilevelFlow
from flows.layers import GlobalRescalingLayer
from flows.distributions import Prior, FreeScalarDistribution, Phi4Dist

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor
Module: TypeAlias = torch.nn.Module
IterableDataset: TypeAlias = torch.utils.data.IterableDataset

logging.getLogger().setLevel("WARNING")


##############
#   LAYERS   #
##############

ADDITIVE_BLOCK = {
    "transform": transforms.PointwiseAdditiveTransform,
    "transform_spec": {},
    "net_spec": {
        "hidden_shape": [4, 4],
        "activation": torch.nn.Tanh(),
        "final_activation": torch.nn.Identity(),
        "use_bias": False,
    },
}
AFFINE_BLOCK = {
    "transform": transforms.PointwiseAffineTransform,
    "transform_spec": {},
    "net_spec": {
        "hidden_shape": [],
        "activation": torch.nn.Tanh(),
        "final_activation": torch.nn.Tanh(),
        "kernel_size": args.ksize,
        "use_bias": False,
    },
}
SPLINE_BLOCK = {
    "transform": transforms.PointwiseRationalQuadraticSplineTransform,
    "transform_spec": {"n_segments": 8, "interval": (-4, 4)},
    "net_spec": {
        "hidden_shape": [4],
        "activation": torch.nn.Tanh(),
        "final_activation": torch.nn.Identity(),
        "use_bias": True,
    },
}

##############
# MODEL SPEC #
##############

MODEL_SPEC = [
    # "upsampling",
    AFFINE_BLOCK,
    "rescaling",
]


##############
# SAVE PARSE #
##############

# Target theory
LATTICE_LENGTH = args.lsize
BETA = args.beta
LAM = args.lam

N_TRAIN = args.nepochs
N_BATCH = args.batch
N_BATCH_VAL = 1000
nsave = args.save


###############
# CREATE DIR. #
###############

wdir_prefix = "L"+str(LATTICE_LENGTH)+"_b"+str(BETA)+"_l"+str(LAM)+"_k"+str(args.ksize)+"_E"+str(N_TRAIN)+"_B"+str(N_BATCH)
wdir_sufix = datetime.today().strftime('_%Y-%m-%d-%H:%M:%S/')
wdir = args.wdir + wdir_prefix + wdir_sufix

logdir = wdir+"logs/"
logfile = logdir + "log.txt"

if os.path.isdir(wdir) == False:
    os.mkdir(wdir)
    os.mkdir(logdir)
# elif args.overwrite == True:
#     shutil.rmtree(wdir) # could be dangerous
#     os.mkdir(wdir)
#     os.mkdir(logdir)
else:
    print("Directory wdir already exists. Overwriting...")
    # raise NotImplementedError("Working directory already exists!")


############################
# LOGS FOR REPRODUCIBILITY #
############################

# Create patch with non-staged changed (from tracked files)
create_gitpatch = "git diff > {}".format(logdir+"gitpatch.path")
os.system(create_gitpatch)

# Print current branch to branchinfo file
os.system("echo Current branch: > {}".format(logdir+"branchinfo.txt"))
os.system("git rev-parse --abbrev-ref HEAD >> {}".format(logdir+"branchinfo.txt"))
os.system("echo  >> {}".format(logdir+"branchinfo.txt"))

# Print current commit of branch to branchinfo file
os.system("echo Current commit: >> {}".format(logdir+"branchinfo.txt"))
os.system("git rev-parse --short HEAD >> {}".format(logdir+"branchinfo.txt"))
os.system("echo  >> {}".format(logdir+"branchinfo.txt"))

# Print current estate of the repository to branchinfo file
os.system("echo Estate of repository: >> {}".format(logdir+"branchinfo.txt"))
os.system("git show-ref >> {}".format(logdir+"branchinfo.txt"))

# Write command used to run the script to the logfile.
with open(logfile, "w") as file_object:
    file_object.write(" ".join(sys.argv)+"\n")
    print(vars(args), file=file_object)
    file_object.write("Pytorch seed: "+str(torch.initial_seed())+"\n")


###############
#  LOAD MODEL #
###############

model = MultilevelFlow(
    beta=BETA,
    lam=LAM,
    model_spec=MODEL_SPEC,
)

# dist = torch.distributions.Normal(
#     loc=torch.zeros((LATTICE_LENGTH, LATTICE_LENGTH)),
#     scale=torch.ones((LATTICE_LENGTH, LATTICE_LENGTH)),
# )
# dist = Phi4Dist(0.576, LAM, 8, thermalization=100, discard=1)
dist = Phi4Dist(0.576, LAM, 16, thermalization=10000, discard=20)
train_dataloader = Prior(dist, sample_shape=[N_BATCH, 1])
val_dataloader = Prior(dist, sample_shape=[N_BATCH_VAL, 1])

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
tb_logger = pl.loggers.TensorBoardLogger(save_dir=wdir)

# check https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing.html#automatic-saving
# check https://forums.pytorchlightning.ai/t/how-to-get-the-checkpoint-path/327/2
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    every_n_train_steps=nsave, # saves every n training steps
    save_top_k=-1, # saves all checkpoints, without overwriting
    save_last=True,
)

trainer = pl.Trainer(
    default_root_dir=wdir,
    gpus=1,
    max_steps=N_TRAIN,  # total number of training steps
    val_check_interval=10,  # how often to run sampling
    limit_val_batches=1,  # one batch for each val step
    callbacks=[lr_monitor, checkpoint_callback],
    enable_checkpointing=True,
)

trainer.validate(model, val_dataloader)

# Save randomized model
# trainer.save_checkpoint(wdir+"lightning_logs/manual_saves/first.ckpt")

trainer.fit(model, train_dataloader, val_dataloader)
