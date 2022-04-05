#############
# Libraries #
#############

from __future__ import annotations
import sys
import os
import shutil

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
        "hidden_shape": [4, 4, 4, 4],
        "activation": torch.nn.Tanh(),
        "final_activation": torch.nn.Tanh(),
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
    AFFINE_BLOCK,
    AFFINE_BLOCK,
    "rescaling",
]


#############
#  PARSING  #
#############

def parse_info(info, c):
    return info.split(c)[1].split("_")[0]

parser = argparse.ArgumentParser()
# python3 main/flow_hmc-check.py -n NTRAJ -t TAU -ns NSTEPS
# python3 main/flow-hmc-check.py -n 1 -t 1.0 -ns 1 -w results/trained_networks/testdir/ --model=models/L8_b0.7_l0.5_model.pth
parser.add_argument("-n", "--ntraj", help="Number of trajectories", type=int, required=True)
parser.add_argument("-t", "--tau", help="HMC trajectory length", default=1.0, type=float)
parser.add_argument("-ns", "--nsteps", help="Number of integration steps", default=10, type=int)
parser.add_argument("-s", "--save", help="Save every s configurations", type=int, required=False, default=1)
parser.add_argument("-m", "--model", help="Path to pytorch model", type=str, required=True)
parser.add_argument("-w", "--wdir", help="Working directory", type=str, required=True)
parser.add_argument("-ow", "--overwrite", help="Working directory", type=bool, required=False, default=False)

args = parser.parse_args()

###############
# CREATE DIR. #
###############

wdir = args.wdir
configdir = wdir+"configs/"
logdir = wdir+"log/"
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
    raise NotImplementedError("Working directory already exists!")

create_gitpatch = "git diff > {}".format(logdir+"gitpatch.path")
os.system(create_gitpatch)

os.system("echo Current branch: > {}".format(logdir+"branchinfo.txt"))
os.system("git rev-parse --abbrev-ref HEAD >> {}".format(logdir+"branchinfo.txt"))
os.system("echo Estate of repository: >> {}".format(logdir+"branchinfo.txt"))
os.system("git show-ref >> {}".format(logdir+"branchinfo.txt"))


###############
#  LOAD MODEL #
###############

model_name = args.model

# Target theory
LATTICE_LENGTH = int(parse_info(model_name, "L"))
BETA = float(parse_info(model_name, "_b"))
LAM = float(parse_info(model_name, "_l"))

# HMC params
tau = args.tau
n_steps = args.nsteps
ntraj = args.ntraj
nsave = args.save

# Load model
model = MultilevelFlow(
    beta=BETA,
    lam=LAM,
    model_spec=MODEL_SPEC,
)

model.load_state_dict(torch.load(args.model))

dist = torch.distributions.Normal(
    loc=torch.zeros((LATTICE_LENGTH, LATTICE_LENGTH)),
    scale=torch.ones((LATTICE_LENGTH, LATTICE_LENGTH)),
)

val_dataloader = Prior(dist, sample_shape=[1, 1])
phi = val_dataloader.sample()

wdir_prefix = "L"+str(LATTICE_LENGTH)+"_b"+str(BETA)+"_l"+str(LAM)


def save_config(phi, path):
    config = phi.detach().numpy().reshape(-1)
    np.savetxt(path, config)


for i in (range(ntraj)):
    flow_hmc(phi, model, tau=tau, n_steps=n_steps, reversibility=False)
    mag_i = phi.mean(axis=(1,2,3)).item()

    with open(mesdir+wdir_prefix+"_mag", "a") as file_object:
        file_object.write(str(mag_i)+"\n")

    if i % nsave == 0:
        save_config(phi, configdir+wdir_prefix+"_n"+str(i))

