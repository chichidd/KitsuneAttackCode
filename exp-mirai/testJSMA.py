'''
script used for testing JSMA on Kitsune model (slow comparing to FGSM and CW).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from PytorchAD import AnomalyDetector
from ADClassifier import ADClassifier
from utils import get_feature_map, load_data
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import time
from advertorch.attacks import JacobianSaliencyMapAttack

def Dist(orig, adv, printInfo=False):
    '''
    print the L0, L1, L2 and L_inf distance between adversarial example and the origin.
    '''
    dif = orig - adv
    res = []
    p_list = [0., 1., 2., float('inf')]
    for p in p_list:
        dist = torch.norm(dif, p, 1)
        res.append(torch.mean(dist))
        if printInfo:
            print("L_{} distance\n {}±{}".format(p, torch.mean(dist), torch.std(dist)))

    return res

# Load data
mirai = load_data("data/mirai_tshark_onhanqiupc.pkl")
labels = load_data("data/mirai_labels.pkl")

# KitNET params
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 50000 #the number of instances used to train the anomaly detector (ensemble itself)

# Train FM and initialize AD
v = get_feature_map(mirai, FMgrace, maxAE)
AD = AnomalyDetector(v, device="cpu")

#AD.load_state_dict(torch.load("AD_model_mirai_state_dict.pt"))
AD = torch.load("model/AD_model_mirai.pt")
AD.eval()

returnLogit = False
threshold = 1.
ADC = ADClassifier(AD, threshold, logit=returnLogit).cpu().eval()


sample_num = 10
# benign samples used for avalability attack
Bsample_idx = np.random.choice(np.where(labels == 0)[0], sample_num, replace=False)
mirai_Bsample = torch.tensor(mirai[Bsample_idx])
label_Bsample = torch.tensor(labels[Bsample_idx])

# malicious samples with score closed to threshold 1
mirai_predM = mirai[AD(torch.tensor(mirai))>threshold]
labels_predM = labels[AD(torch.tensor(mirai))>threshold]
_, Msample_idx = torch.topk(AD(torch.tensor(mirai_predM))-threshold, k=sample_num,largest=False)
mirai_Msample = torch.tensor(mirai_predM[Msample_idx])
label_Msample = torch.tensor(labels_predM[Msample_idx])

print("Sample good")
theta = 10
adversary = JacobianSaliencyMapAttack(ADC, num_classes=2, clip_min=mirai_Bsample.min().item(), clip_max=mirai_Bsample.max().item(), theta=theta)
adv_FP = adversary.perturb(mirai_Bsample, label_Msample.reshape(-1).long())

adversary = JacobianSaliencyMapAttack(ADC, num_classes=2, clip_min=mirai_Msample.min().item(), clip_max=mirai_Msample.max().item(), theta=theta)
adv_FN = adversary.perturb(mirai_Msample, label_Bsample.reshape(-1).long())

print(torch.argmax(ADC(mirai_Bsample),dim=1))

print(torch.argmax(ADC(adv_FP),dim=1))

print(torch.argmax(ADC(mirai_Msample),dim=1))

print(torch.argmax(ADC(adv_FN), dim=1))

Dist(adv_FP, mirai_Bsample, True)

Dist(adv_FN, mirai_Msample,  True)