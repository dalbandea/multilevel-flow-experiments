#############
# Libraries #
#############

from __future__ import annotations
import sys
import os
import shutil
from datetime import datetime

# sys.path.append("../../")
sys.path.append(".")

import math
import logging

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import argparse
from jsonargparse.typing import PositiveInt, PositiveFloat, NonNegativeFloat

import flows.phi_four as phi_four
import flows.transforms as transforms
import flows.utils as utils
from flows.flow_hmc import *
from flows.models import MultilevelFlow
from flows.layers import GlobalRescalingLayer
from flows.distributions import Prior, FreeScalarDistribution

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor
Module: TypeAlias = torch.nn.Module
IterableDataset: TypeAlias = torch.utils.data.IterableDataset

logging.getLogger().setLevel("WARNING")



#############
#  PARSING  #
#############

parser = argparse.ArgumentParser()
# python3 main/flow_hmc-check.py -n NTRAJ -t TAU -ns NSTEPS
# python3 main/flow-hmc-check.py -n 1 -L 8 -t 1.0 -ns 10 --model=models/L8_b0.7_l0.5_model.ckpt
parser.add_argument("-L", "--lsize", help="Lattice size", type=int, required=True)
parser.add_argument("-n", "--ntraj", help="Number of trajectories", type=int, required=True)
parser.add_argument("-t", "--tau", help="HMC trajectory length", default=1.0, type=float)
parser.add_argument("-ns", "--nsteps", help="Number of integration steps", default=10, type=int)
parser.add_argument("-s", "--save", help="Save every s configurations", type=int, required=False, default=1)
parser.add_argument("-m", "--model", help="Path to pytorch model", type=str, required=True)
parser.add_argument("-w", "--wdir", help="Working directory", type=str, default="results/flow_hmc/")
parser.add_argument("-ow", "--overwrite", help="Working directory", type=bool, required=False, default=False)

args = parser.parse_args()


###############################
#  LOAD MODEL AND HMC PARAMS  #
###############################

model_path = args.model

# Load model
model = MultilevelFlow.load_from_checkpoint(model_path)
model.eval()

# Target theory
LATTICE_LENGTH = args.lsize
BETA = model.hparams.beta
LAM = model.hparams.lam
wdir_prefix = "L"+str(LATTICE_LENGTH)+"_b"+str(BETA)+"_l"+str(LAM)
wdir_sufix = datetime.today().strftime('_%Y-%m-%d-%H:%M:%S/')

# HMC params
tau = args.tau
n_steps = args.nsteps
ntraj = args.ntraj
nsave = args.save


###############
# CREATE DIR. #
###############


wdir = args.wdir + wdir_prefix + wdir_sufix
configdir = wdir+"configs/"
logdir = wdir+"logs/"
logfile = logdir + "log.txt"
mesdir = wdir+"measurements/"

if os.path.isdir(wdir) == False:
    os.mkdir(wdir)
    os.mkdir(configdir)
    os.mkdir(logdir)
    os.mkdir(mesdir)
# elif args.overwrite == True:
#     shutil.rmtree(wdir) # could be dangerous
#     os.mkdir(wdir)
#     os.mkdir(configdir)
#     os.mkdir(logdir)
#     os.mkdir(mesdir)
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


#######
# HMC #
#######

# Initial sample from normal distribution
dist = torch.distributions.Normal(
    loc=torch.zeros((LATTICE_LENGTH, LATTICE_LENGTH)),
    scale=torch.ones((LATTICE_LENGTH, LATTICE_LENGTH)),
)
val_dataloader = Prior(dist, sample_shape=[1, 1])
phi = val_dataloader.sample()


# Function to save configurations
def save_config(phi, path):
    config = phi.detach().numpy().reshape(-1)
    np.savetxt(path, config)


# Perform HMC
for i in (range(ntraj)):
    flow_hmc(phi, model, tau=tau, n_steps=n_steps, reversibility=False)
    mag_i = phi.mean(axis=(1,2,3)).item()

    if i == 0:
        with open(mesdir+wdir_prefix+"_mag.txt", "w") as file_object:
            file_object.write(str(mag_i)+"\n")

    with open(mesdir+wdir_prefix+"_mag.txt", "a") as file_object:
        file_object.write(str(mag_i)+"\n")

    if i % nsave == 0:
        save_config(phi, configdir+wdir_prefix+"_n"+str(i))

