from FeatureExtractor import *
from KitNET.KitNET import KitNET

class Kitsune:
    def __init__(self,max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75,):
        #init packet feature extractor (AfterImage)
        self.FE = FE()

        #init Kitnet
        self.AnomDetector = KitNET(self.FE.get_num_features(),max_autoencoder_size,FM_grace_period,AD_grace_period,learning_rate,hidden_ratio)

    def proc_next_packet(self, modify=False, p=0.0):
        # create feature vector
        x = self.FE.get_next_vector(modify, p)
        if len(x) == 0:
            return -1 # Error or no packets left
        if np.all(x==0):
            return 0
        # process KitNET
        return self.AnomDetector.process(x) # will train during the grace periods, then execute on all the rest.

