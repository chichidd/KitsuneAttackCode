import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=None, corruption_level=0.0, device="cpu"):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        # in the paper, the default hidden ratio is 0.75
        if hidden_size is None:
            self.hidden_size = int(np.ceil(input_size * 0.75))
        else:
            self.hidden_size = hidden_size

        self.device = device

        # initialize the weight of matrix, same random state 1234
        a = 1. / input_size
        self.rng = np.random.RandomState(1234)
        self.weight = nn.Parameter(torch.tensor(self.rng.uniform(  # initialize W uniformly
            low=-a,
            high=a,
            size=(self.input_size, self.hidden_size)), device=self.device))

        self.hbias = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float64, device=self.device))
        self.vbias = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float64, device=self.device))

        # in the source code of Kitsune, it is corruption level
        self.corruption_level = corruption_level

        # used for normalization (put value into 0 and 1)
        self.norm_max = nn.Parameter(torch.ones((self.input_size,), dtype=torch.float64, device=self.device) * torch._six.inf * (-1),
                                     requires_grad=False)
        self.norm_min = nn.Parameter(torch.ones((self.input_size,), dtype=torch.float64, device=self.device) * torch._six.inf,
                                     requires_grad=False)

    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1

        return torch.tensor(self.rng.binomial(size=tuple(input.shape),
                                              n=1,
                                              p=1 - corruption_level), device=self.device) * input

    def get_hidden_layer(self, x):

        return torch.sigmoid(F.linear(x, weight=torch.transpose(self.weight, 0, 1), bias=self.hbias))

    def get_reconstructed_x(self, x):
        return torch.sigmoid(F.linear(self.get_hidden_layer(x), weight=self.weight, bias=self.vbias))

    def forward(self, x):

        if self.training:
            # during traing, max_iteration = 1:
            x = x.reshape(-1)
            # update norms

            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]

            # 0-1 normalize
            x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)

            if self.corruption_level > 0:
                x = self.get_corrupted_input(x, self.corruption_level)

            reconstructed_x = self.get_reconstructed_x(x)
            return reconstructed_x.reshape(1, -1), x.reshape(1, -1)
        else:
            # 0-1 normalize

            x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)

            reconstructed_x = self.get_reconstructed_x(x)

            return reconstructed_x, x