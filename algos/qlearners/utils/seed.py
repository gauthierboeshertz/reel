import torch
import numpy 
import random

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.random.manual_seed(seed)
