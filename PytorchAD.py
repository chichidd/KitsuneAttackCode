import torch
import torch.nn as nn
from PytorchAutoencoder import Autoencoder
import numpy as np

class AnomalyDetector(nn.Module):
    def __init__(self, v, corruption_level=0, hidden_ratio=0.75, device="cpu"):
        '''
        n: the number of features in your input dataset (i.e., x \in R^n)
        max_autoencoder_size: the maximum size of any autoencoder in the ensemble layer
        hidden_ratio: the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
        '''
        super(AnomalyDetector, self).__init__()
        self.v = v
        self.ensemble_layer = nn.ModuleList([])
        self.device = device
        for vi in v:
            self.ensemble_layer.append(Autoencoder(input_size=len(vi), hidden_size=int(np.ceil(hidden_ratio * len(vi))),
                                                   corruption_level=corruption_level, device=self.device))

        self.output_layer = Autoencoder(input_size=len(v), hidden_size=int(np.ceil(hidden_ratio * len(v))),
                                        corruption_level=corruption_level, device=self.device)

        # phi: to save the highest anomaly score during training.
        self.phi = torch.tensor(0, device=self.device)

        # intermediate representation
        self.S_l1 = torch.zeros(len(v), device=self.device, requires_grad=True)

    def forward(self, x):
        # Attention: the batch size is 1, so the shape of x should be 1 \times n_dim    

        # generate RMSE of each autoencoder
        z_list = []

        for i, AE in enumerate(self.ensemble_layer):
            reconstructed_xi, normalized_xi = AE(x[:, self.v[i]])
            z_list.append(torch.sqrt(torch.mean((reconstructed_xi - normalized_xi) ** 2, dim=1)).reshape(-1, 1))

        # generate input of output_layer
        self.S_l1 = torch.cat(z_list, dim=1)

        if self.training:

            # during training, each autoencoder is trained using the error between its own output and input
            reconstructed_S_l1, normalized_S_l1 = self.output_layer(self.S_l1.detach())
            score = torch.sqrt(torch.mean((reconstructed_S_l1 - normalized_S_l1) ** 2))
            z_list.append(score)
            if score.item() > self.phi:
                self.phi = score.item()
            return z_list

        else:
            reconstructed_S_l1, normalized_S_l1 = self.output_layer(self.S_l1)
            score = torch.sqrt(torch.mean((reconstructed_S_l1 - normalized_S_l1) ** 2, dim=1))
            return score