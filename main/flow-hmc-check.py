from __future__ import annotations
import argparse

#############
#  PARSING  #
#############

parser = argparse.ArgumentParser()
# python3 main/flow_hmc-check.py -n NTRAJ -t TAU -ns NSTEPS
# python3 main/flow-hmc-check.py -n 1 -L 8 -t 1.0 -ns 10 --model=models/L8_b0.7_l0.5_model.ckpt
# wdir must end with /, otherwise a mess will happen. TODO: correct this
parser.add_argument("-L", "--lsize", help="Lattice size", type=int, required=True)
parser.add_argument("-b", "--beta", help="Beta value (from model if not provided)", type=float)
parser.add_argument("-n", "--ntraj", help="Number of trajectories", type=int, required=True)
parser.add_argument("-t", "--tau", help="HMC trajectory length", default=1.0, type=float)
parser.add_argument("-ns", "--nsteps", help="Number of integration steps", default=10, type=int)
parser.add_argument("-s", "--save", help="Save every s configurations", type=int, required=False, default=1)
parser.add_argument("--seed", help="Set torch seed", type=int, required=False)
parser.add_argument("-m", "--model", help="Path to pytorch model", type=str, required=True)
parser.add_argument("-w", "--wdir", help="Working directory", type=str, default="results/flow_hmc/")
parser.add_argument("-T", "--tag", help="Tag", type=str, required = False, default = "")
parser.add_argument("--replica", help="Replica number. If !=0, wdir must point to existing directory with existing replica 0", type=int, required=False, default = 0)
parser.add_argument("-ow", "--overwrite", help="Working directory", type=bool, required=False, default=False)

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
tau = args.tau
n_steps = args.nsteps
ntraj = args.ntraj
nsave = args.save

###############
# CREATE DIR. #
###############

wdir_prefix = "L"+str(LATTICE_LENGTH)+"_b"+str(BETA)+"_l"+str(LAM)+"_T"+str(args.tag)
wdir_sufix = datetime.today().strftime('_%Y-%m-%d-%H:%M:%S/')
replica = args.replica

if replica == 0:
    wdir = args.wdir + wdir_prefix + wdir_sufix + "0-r0/"
elif replica > 0:
    if os.path.isdir(args.wdir+"0-r0/"):
        wdir = args.wdir + str(replica) + "-r" + str(replica) + "/"
    else:
        raise NotImplementedError("Replica 0 folder not found")
else:
    raise NotImplementedError("Replica number not valid")

configdir = wdir+"configs/"
logdir = wdir+"logs/"
logfile = logdir + "log.txt"
mesdir = wdir+"measurements/"

if os.path.isdir(wdir) == False:
    os.makedirs(wdir)
    os.mkdir(configdir)
    os.mkdir(logdir)
    os.mkdir(mesdir)
# elif args.overwrite == True:
#     shutil.rmtree(wdir) # could be dangerous
#     os.mkdir(wdir)
#     os.mkdir(configdir)
#     os.mkdir(logdir)
#     os.mkdir(mesdir)
elif replica > 0:
    os.mkdir(wdir)
    os.mkdir(configdir)
    os.mkdir(logdir)
    os.mkdir(mesdir)
else:
    print("Directory wdir already exists.")
    if args.overwrite == False:
        raise NotImplementedError("Overwrite set to False and directory already exists")
    elif args.overwrite == True:
        print("Overwriting set to True. Overwriting...")
    


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


if model.n_upsampling == 1:
    model_flow = MultilevelFlow.load_from_checkpoint(model_path)
    model_flow.eval()
    layers_flow = [model_flow.get_submodule("flow.1"), model_flow.get_submodule("flow.2")]
    model_flow.flow = utils.Flow(*layers_flow)
    model_flow.action = phi_four.PhiFourActionBeta(BETA, LAM)
    val_dataloader = Prior(dist, sample_shape=[4, 1])
    phi = val_dataloader.sample()
    phi = model.flow(phi)[0].detach()
    model.flow = model_flow.flow


# Perform HMC
for i in (range(ntraj)):
    dH = flow_hmc(phi, model, tau=tau, n_steps=n_steps, reversibility=False)
    mag_i = magnetization(phi)
    susc_i = susceptibility(phi)
    phi_t = he_flow(phi, 1.0)
    susc_it = susceptibility(phi_t)

    with open(mesdir+wdir_prefix+"_mag.txt", "a") as file_object:
        file_object.write(str(mag_i)+"\n")

    with open(mesdir+wdir_prefix+"_susc.txt", "a") as file_object:
        file_object.write(str(susc_i)+"\n")

    with open(mesdir+wdir_prefix+"_susct.txt", "a") as file_object:
        file_object.write(str(susc_it)+"\n")

    with open(mesdir+wdir_prefix+"_expdH.txt", "a") as file_object:
        file_object.write(str(dH)+"\n")

    if i % nsave == 0:
        save_config(phi, configdir+wdir_prefix+"_n"+str(i))



M = np.loadtxt(mesdir+wdir_prefix+"_mag.txt")
acc = 1-np.mean(np.roll(M,-1) == M)
print(acc)
