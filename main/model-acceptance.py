from __future__ import annotations
import argparse

#############
#  PARSING  #
#############

parser = argparse.ArgumentParser()
# python3 main/flow_hmc-check.py -n NTRAJ -t TAU -ns NSTEPS
# python3 main/model-acceptance.py -n 1 -L 8 --model=models/L8_b0.7_l0.5_model.ckpt --wdir=
# wdir must end with /, otherwise a mess will happen. TODO: correct this
parser.add_argument("-L", "--lsize", help="Lattice size", type=int, required=True)
parser.add_argument("-b", "--beta", help="Beta value (from model if not provided)", type=float)
parser.add_argument("-n", "--ntraj", help="Number of trajectories", type=int, required=True)
parser.add_argument("--seed", help="Set torch seed", type=int, required=False)
parser.add_argument("-m", "--model", help="Path to pytorch model", type=str, required=True)
parser.add_argument("-w", "--wdir", help="Working directory containing model folder", type=str, required=True)

args = parser.parse_args()

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
import numpy as np

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
from flows.distributions import Prior, FreeScalarDistribution
from flows.measurements import magnetization, susceptibility, he_flow, he_flow_p

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor
Module: TypeAlias = torch.nn.Module
IterableDataset: TypeAlias = torch.utils.data.IterableDataset

logging.getLogger().setLevel("WARNING")


###############################
#  LOAD MODEL AND HMC PARAMS  #
###############################

model_path = args.model

# Load model
model = MultilevelFlow.load_from_checkpoint(model_path)
model.eval()

# Target theory
LATTICE_LENGTH = args.lsize
LAM = model.hparams.lam
if args.beta == None:
    BETA = model.hparams.beta
else:
    BETA = args.beta
    # Change action of the model
    model.action = phi_four.PhiFourActionBeta(BETA, LAM)

# HMC params
ntraj = args.ntraj


###############
# CREATE DIR. #
###############

wdir_prefix = "L"+str(LATTICE_LENGTH)+"_b"+str(BETA)+"_l"+str(LAM)
wdir_sufix = datetime.today().strftime('_%Y-%m-%d-%H:%M:%S/')

if os.path.isdir(args.wdir):
    acc_dir = os.path.join(args.wdir, "MH-acceptance")
    if not os.path.isdir(acc_dir):
        os.mkdir(acc_dir)
else:
    raise NotImplementedError("Model directory does not exist")

wdir = os.path.join(acc_dir, wdir_prefix+wdir_sufix)

if os.path.isdir(wdir):
    raise NotImplementedError("Acceptance wdir already exists")
else:
    os.mkdir(wdir)

acc_file = os.path.join(wdir, "acc-history.txt")
logdir = os.path.join(wdir, "logs")
os.mkdir(logdir)
logfile = os.path.join(logdir, "log.txt")


############################
# LOGS FOR REPRODUCIBILITY #
############################

# Create patch with non-staged changed (from tracked files)
create_gitpatch = "git diff > {}".format(os.path.join(logdir,"gitpatch.path"))
os.system(create_gitpatch)

branchinfofile = os.path.join(logdir,"branchinfo.txt")

# Print current branch to branchinfo file
os.system("echo Current branch: > {}".format(branchinfofile))
os.system("git rev-parse --abbrev-ref HEAD >> {}".format(branchinfofile))
os.system("echo  >> {}".format(branchinfofile))



# Print current commit of branch to branchinfo file
os.system("echo Current commit: >> {}".format(branchinfofile))
os.system("git rev-parse --short HEAD >> {}".format(branchinfofile))
os.system("echo  >> {}".format(branchinfofile))

# Print current estate of the repository to branchinfo file
os.system("echo Estate of repository: >> {}".format(branchinfofile))
os.system("git show-ref >> {}".format(branchinfofile))


# Write command used to run the script to the logfile.
with open(logfile, "w") as file_object:
    file_object.write(" ".join(sys.argv)+"\n")
    print(vars(args), file=file_object)
    file_object.write("Pytorch seed: "+str(torch.initial_seed())+"\n")


#################
# MH ACCEPTANCE #
#################

# Initial sample from normal distribution
dist = torch.distributions.Normal(
    loc=torch.zeros((LATTICE_LENGTH, LATTICE_LENGTH)),
    scale=torch.ones((LATTICE_LENGTH, LATTICE_LENGTH)),
)

val_dataloader = Prior(dist, sample_shape=[1, 1])

def markov_chain(acc_file, it, model, val_dataloader):
    phi_curr, weight_curr = model.forward(model.sample(val_dataloader))
    
    for i in range(it):
        phi_prop, weight_prop = model.forward(model.sample(val_dataloader))
        if torch.rand(1).item() < min(1, math.exp(weight_curr - weight_prop)):
            weight_curr = weight_prop
            with open(acc_file, "a") as file_object:
                file_object.write("1\n")
        else:
            with open(acc_file, "a") as file_object:
                file_object.write("0\n")

markov_chain(acc_file, ntraj, model, val_dataloader)
