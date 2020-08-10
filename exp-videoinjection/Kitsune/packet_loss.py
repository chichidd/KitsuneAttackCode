from Kitsune import Kitsune
import numpy as np
import time
import pickle

# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 100000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 1000000 #the number of instances used to train the anomaly detector (ensemble itself)

# Build Kitsune
K = Kitsune(maxAE,FMgrace,ADgrace)

print("Running Kitsune:")
RMSEs = []
i = 0
start = time.time()
# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
modify = False
packet_loss_proba = 0.3
while True:
    i+=1
    if i % 10000 == 0:
        print(i)
    if i > 1500000 and i < 1800000:
        modify = True
    else:
        modify = False
    rmse = K.proc_next_packet(modify, packet_loss_proba)

    if rmse == -1:
        break
    RMSEs.append(rmse)
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))

pickle.dump(RMSEs, open("RMSEs_packet_loss_{}.pkl".format(packet_loss_proba), "wb"))

