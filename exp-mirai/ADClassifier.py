import torch
import torch.nn as nn
# create AD classifier
class ADClassifier(nn.Module):
    def __init__(self, AD, th, logit=False):
        super(ADClassifier, self).__init__()
        self.AD = AD
        self.threshold = th
        self.logit = logit
    def forward(self, x):
        '''
        score = torch.cat([self.AD(x).reshape(-1,1), torch.ones((x.shape[0],1), dtype=torch.float64)*self.threshold], dim=1)

        if self.logit:
            return -score
        else:
            return torch.softmax(-score, dim=1) # the sign "-" makes the model classifies the instances with error lower than self.threshold as benign
        '''

        # Sigmoid method
        # If AD's score bigger than threshold, the logit > 0 and the sigmoid value > 0.5.
        logit = self.AD(x).reshape(-1,1)-self.threshold
        if self.logit:
            return logit
        else:
            return torch.cat([1 - torch.sigmoid(logit), torch.sigmoid(logit)], dim=1)
