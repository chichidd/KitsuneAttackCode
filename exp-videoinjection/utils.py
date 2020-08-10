from Kitsune.KitNET.corClust import corClust
import pickle


def get_feature_map(alldata, FM_grace_period, max_autoencoder_size):
    '''
    generate feature map of AD
    '''
    FM = corClust(alldata.shape[1])
    for i in range(FM_grace_period):
        FM.update(alldata[i])
    return FM.cluster(max_autoencoder_size)


def load_data(path="mirai_np.pkl"):
    '''
    Load processed numpy array of mirai dataset by the origin FE.
    '''
    return pickle.load(open(path, "rb"))
