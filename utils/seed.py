import random
import torch


def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
