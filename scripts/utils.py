import random
from torch.autograd import Variable
import torch
from types import SimpleNamespace
import _pickle as cPickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def save_pickle(f, obj):
    with open(f, "wb+") as fs:
        cPickle.dump(obj, fs)


def load_pickle(f):
    with open(f, "rb") as fs:
        return cPickle.load(fs)


def plot_losses(ldict):
    losses = pd.Dataframe(ldict)
    _, ax = plt.subplots(figsize=(30, 10))
    plot = sns.lineplot(data=losses, ax=ax)
    plot.figure.savefig("/content/drive/MyDrive/Night2Day/checkpoints_2/losses.jpg")


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer. Please provide a valid `max_size`"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


opt = {
    "epoch": 0,
    "n_epochs": 30,
    "ds_name": "Night2Day",
    "batch_size": 16,
    "lr": 0.002,
    "b1": 0.999,
    "b2": 0.999,
    "decay_epoch": 10,
    "H": 300,
    "W": 300,
    "C": 3,
    "sample_interval": 300,
    "checkpoint_interval": 5,
    "lambda_cyc": 11.0,
    "lambda_id": 5.5
}
opt = SimpleNamespace(**opt)
