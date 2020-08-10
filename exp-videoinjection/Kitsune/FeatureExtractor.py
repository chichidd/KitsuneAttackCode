#Check if cython code has been compiled
import os
import subprocess
print("Importing AfterImage Cython Library")
if not os.path.isfile("AfterImage.c"): #has not yet been compiled, so try to do so...
    cmd = "python setup.py build_ext --inplace"
    subprocess.call(cmd,shell=True)
#Import dependencies
import netStat as ns
import csv
import numpy as np
print("Importing Scapy Library")
from scapy.all import *
import os.path
import platform
import subprocess
import pickle

class FE:
    def __init__(self,):
        
        self.curPacketIndx = 0
        self.packages = pickle.load(open("../data/videoinj_packages.pkl", "rb"))
        ### Prep Feature extractor (AfterImage) ###
        maxHost = 100000000000
        maxSess = 100000000000
        self.nstat = ns.netStat(np.nan, maxHost, maxSess)


    def get_next_vector(self, modify=False, p=0.0):
        '''
        p is the packet loss probability
        '''
        if self.curPacketIndx >= len(self.packages):
            return []
        IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp = self.packages[self.curPacketIndx]
        self.curPacketIndx = self.curPacketIndx + 1


        ### Packet loss
        try:

            if modify:
                
                if np.random.rand()<p:
                    return np.zeros(100)

            return self.nstat.updateGetStats(IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol,
                                             datagramSize,
                                             timestamp)
        except Exception as e:
            print(e)
            return []


    def get_num_features(self):
        return len(self.nstat.getNetStatHeaders())



