import numpy as np
import torch
import torch.nn.functional as F

'''
Pytorch version of FE (no sqrt operation)
'''

class incStat:
    def __init__(self, Lambda, ID, init_time=0, isTypeDiff=False):  # timestamp is creation time
        self.ID = ID
        self.Lambda = Lambda  # Decay Factor
        self.CF1 = torch.zeros(len(self.Lambda))  # linear sum
        self.CF2 = torch.zeros(len(self.Lambda))  # sum of squares
        self.w = 1e-20 * torch.ones(len(self.Lambda)) # weight
        self.isTypeDiff = isTypeDiff
        self.lastTimestamp = init_time
        self.cur_mean = torch.zeros(1)
        self.cur_var = torch.zeros(1)
        self.cur_std = torch.zeros(1)
        self.covs = []  # a list of incStat_covs (references) with relate to this incStat

    def insert(self, v, t=0):  # v is a scalar, t is v's arrival the timestamp
        if self.isTypeDiff:
            v = F.relu(t - self.lastTimestamp)

        self.processDecay(t)

        # update with v
        self.CF1 += v
        self.CF2 += torch.pow(v, 2)
        self.w += 1.
        self.cur_mean = torch.zeros(1)
        self.cur_var = torch.zeros(1)
        self.cur_std = torch.zeros(1)

        # update covs (if any)
        for cov in self.covs:
            cov.update_cov(self.ID, v, t)

    def processDecay(self, timestamp):
        factor = torch.pow(2, (-self.Lambda * F.relu(timestamp - self.lastTimestamp)))
        self.CF1 = self.CF1 * factor
        self.CF2 = self.CF2 * factor
        self.w = self.w * factor
        if timestamp > self.lastTimestamp:
            self.lastTimestamp = timestamp
        return factor

    def mean(self):
        if len(self.cur_mean) == 1:
            # recompute
            self.cur_mean = self.CF1 / self.w
        return self.cur_mean

    def var(self):
        if len(self.cur_var) == 1:
            # recompute
            self.cur_var = torch.abs(self.CF2 / self.w - self.mean()**2)
        return self.cur_var

    def std(self):
        if len(self.cur_std) == 1:
            self.cur_std = torch.sqrt(self.var())
        return self.cur_std

    def radius(self, other_incStat):  # the radius of a set of incStats

        #return torch.sqrt(self.var() + other_incStat.var() + 1e-8)
        return self.var() + other_incStat.var()

    def magnitude(self, other_incStat):  # the magnitude of a set of incStats

        return self.mean()**2 + other_incStat.mean()**2

    # calculates and pulls all stats on this stream
    def allstats_1D(self, stack=True):
        if stack:
            return torch.stack([self.w, self.mean(), self.var()]).T.reshape(-1)
        else:
            return self.w, self.mean(), self.var()


# like incStat, but maintains stats between two streams
class incStat_cov:
    def __init__(self, incS1, incS2, init_time=0):
        # store references to the streams' incStats
        self.incStats = [incS1, incS2]
        self.lastRes = [0, 0]

        # init sum product residuals
        self.CF3 = torch.zeros(len(incS1.Lambda))  # sum of residue products (A-uA)(B-uB)
        self.w3 = 1e-20 * torch.ones(len(incS1.Lambda))
        self.lastTimestamp_cf3 = init_time

    # other_incS_decay is the decay factor of the other incstat
    # ID: the stream ID which produced (v,t)
    def update_cov(self, ID, v, t):
        # it is assumes that incStat "ID" has ALREADY been updated with (t,v) [this si performed automatically in method incStat.insert()]
        # above is checked by the current condition of usage of this function
        # find incStat
        if ID == self.incStats[0].ID:
            inc = 0
        elif ID == self.incStats[1].ID:
            inc = 1
        else:
            print("update_cov ID error")
            return  ## error

        # Decay other incStat
        self.incStats[not (inc)].processDecay(t)

        # Decay residules
        self.processDecay(t, inc)

        # Compute and update residule
        res = (v - self.incStats[inc].mean())
        resid = (v - self.incStats[inc].mean()) * self.lastRes[not (inc)]

        self.CF3 += resid
        self.w3 += 1.
        self.lastRes[inc] = res

    def processDecay(self, t, micro_inc_indx):

        factor = torch.pow(2, (- (self.incStats[micro_inc_indx].Lambda) * F.relu(t - self.lastTimestamp_cf3)))
        self.CF3 *= factor
        self.w3 *= factor
        self.lastRes[micro_inc_indx] *= factor
        if t > self.lastTimestamp_cf3:
            self.lastTimestamp_cf3 = t
        return factor

    def cov(self):
        return self.CF3 / self.w3

    # Pearson corl. coef
    def pcc(self):
        # ss = self.incStats[0].var() * self.incStats[1].var()
        # cov = self.cov()
        # res = torch.sign(cov) * torch.pow(cov, 2) / ss # reserve sign
        # # replace nan and inf by 0.
        # res[res == float('inf')] = 0.
        # res[res == -float('inf')] = 0.
        # res[res != res] = 0.
        # return res
        #### update 24/06/2020
        div = self.incStats[0].var() * self.incStats[1].var()
        cov = self.cov()
        pcc = torch.sign(cov) * torch.pow(cov, 2)
        pcc[div != 0] = pcc[div != 0] / div[div != 0]
        pcc[div == 0] *= 0

        return pcc

    # calculates and pulls all correlative stats AND 2D stats from both streams (incStat)
    def get_stats2(self):
        return self.incStats[0].radius(self.incStats[1]),\
               self.incStats[0].magnitude(self.incStats[1]),\
               self.cov(),\
               self.pcc()


class incStatDB:
    # default_lambda: use this as the lambda for all streams. If not specified, then you must supply a Lambda with every query.
    def __init__(self, limit=torch.tensor(float('inf')), Lambdas=torch.tensor([5., 3., 1., .1, .01])):
        self.HT = dict()
        self.limit = limit
        self.Lambdas = Lambdas

    # Registers a new stream. init_time: init lastTimestamp of the incStat
    def register(self, ID, init_time, isTypeDiff=False, printNew=False):

        incS = self.HT.get(ID)
        if incS is None:
            if printNew:
                print("New incstat ID:", ID)
            if len(self.HT) + 1 > self.limit:
                raise LookupError(
                    'Adding Entry:\n' + ID + '\nwould exceed incStatHT 1D limit of ' + str(
                        self.limit) + '.\nObservation Rejected.')
            incS = incStat(self.Lambdas, ID, init_time, isTypeDiff)
            self.HT[ID] = incS
        return incS

    # Registers covariance tracking for two streams, registers missing streams
    def register_cov(self, incS1, ID2, init_time, isTypeDiff=False ,printNew=False):

        incS2 = self.register(ID2, init_time, isTypeDiff, printNew)

        # check for pre-exiting link
        for cov in incS1.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return cov
        if printNew:
            print("New inc_COV ", incS1.ID, " and ", incS2.ID)
        # Link incStats
        inc_cov = incStat_cov(incS1, incS2, init_time)
        incS1.covs.append(inc_cov)
        incS2.covs.append(inc_cov)
        return inc_cov

    # Updates and then pulls current 1D stats from the given ID. Automatically registers previously unknown stream IDs
    def update_get_1D_Stats(self, ID, t, v, isTypeDiff=False):
        incS = self.register(ID, t, isTypeDiff)
        incS.insert(v, t)
        return incS.allstats_1D()

    # Updates and then pulls current 1D and 2D stats from the given IDs. Automatically registers previously unknown stream IDs
    def update_get_1D2D_Stats(self, ID1, ID2, t1, v1, printNew=False):  # weight, mean, std

        incS1 = self.register(ID1, t1, printNew=printNew)
        incS1.insert(v1, t1)
        # retrieve/add cov tracker
        inc_cov = self.register_cov(incS1, ID2, t1, printNew=printNew)
        # Update cov tracker
        inc_cov.update_cov(ID1, v1, t1)

        return torch.stack(incS1.allstats_1D(stack=False) + inc_cov.get_stats2()).T.reshape(-1)


'''
Original version
'''
#
# class incStat:
#     def __init__(self, Lambda, ID, init_time=0, isTypeDiff=False):  # timestamp is creation time
#         self.ID = ID
#         self.CF1 = torch.tensor(0, dtype=torch.float64)  # linear sum
#         self.CF2 = torch.tensor(0, dtype=torch.float64)  # sum of squares
#         self.w = torch.tensor(1e-20, dtype=torch.float64)  # weight
#         self.isTypeDiff = isTypeDiff
#         self.Lambda = Lambda  # Decay Factor
#         self.lastTimestamp = init_time
#         self.cur_mean = torch.tensor(float('nan'))
#         self.cur_var = torch.tensor(float('nan'))
#         self.cur_std = torch.tensor(float('nan'))
#         self.covs = []  # a list of incStat_covs (references) with relate to this incStat
#
#     def insert(self, v, t=0):  # v is a scalar, t is v's arrival the timestamp
#         if self.isTypeDiff:
#             v = F.relu(t - self.lastTimestamp)
#
#         self.processDecay(t)
#
#         # update with v
#         self.CF1 += v
#         self.CF2 += torch.pow(v, 2)
#         self.w += 1
#         self.cur_mean = torch.tensor(float('nan'))
#         self.cur_var = torch.tensor(float('nan'))
#         self.cur_std = torch.tensor(float('nan'))
#
#         # update covs (if any)
#         for cov in self.covs:
#             cov.update_cov(self.ID, v, t)
#
#     def processDecay(self, timestamp):
#         factor = torch.pow(2, (-self.Lambda * F.relu(timestamp - self.lastTimestamp)))
#         self.CF1 = self.CF1 * factor
#         self.CF2 = self.CF2 * factor
#         self.w = self.w * factor
#         self.lastTimestamp = timestamp
#         return factor
#
#     def mean(self):
#         if self.cur_mean != self.cur_mean:  # calculate it only once when necessary
#             self.cur_mean = self.CF1 / self.w
#         return self.cur_mean
#
#     def var(self):
#         if self.cur_var != self.cur_var:  # calculate it only once when necessary
#             self.cur_var = torch.abs(self.CF2 / self.w - torch.pow(self.mean(), 2))
#         return self.cur_var
#
#     def std(self):
#         if self.cur_std != self.cur_std:  # calculate it only once when necessary
#             self.cur_std = torch.sqrt(self.var() + 1e-8 ) # + 1e-8 to resolve grad problem
#         return self.cur_std
#
#     def radius(self, other_incStats):  # the radius of a set of incStats
#         A = self.var()
#         for incS in other_incStats:
#             A += incS.var()
#         return torch.sqrt(A + 1e-8)
#
#     def magnitude(self, other_incStats):  # the magnitude of a set of incStats
#         A = torch.pow(self.mean(), 2)
#         for incS in other_incStats:
#             A += torch.pow(incS.mean(), 2)
#         return torch.sqrt(A + 1e-8)
#
#     # calculates and pulls all stats on this stream
#     def allstats_1D(self, ):
#         mean = self.mean()
#         var = self.var()
#         std = self.std()
#         return torch.cat([self.w.reshape(1, -1), mean.reshape(1, -1), var.reshape(1, -1)])
#
#
# # like incStat, but maintains stats between two streams
# class incStat_cov:
#     def __init__(self, incS1, incS2, init_time=0):
#         # store references to the streams' incStats
#         self.incStats = [incS1, incS2]
#         self.lastRes = [0, 0]
#
#         # init sum product residuals
#         self.CF3 = torch.tensor(0.).double()  # sum of residue products (A-uA)(B-uB)
#         self.w3 = torch.tensor(1e-20).double()
#         self.lastTimestamp_cf3 = init_time
#
#     # other_incS_decay is the decay factor of the other incstat
#     # ID: the stream ID which produced (v,t)
#     def update_cov(self, ID, v, t):
#         # it is assumes that incStat "ID" has ALREADY been updated with (t,v) [this si performed automatically in method incStat.insert()]
#         # above is checked by the current condition of usage of this function
#         # find incStat
#         if ID == self.incStats[0].ID:
#             inc = 0
#         elif ID == self.incStats[1].ID:
#             inc = 1
#         else:
#             print("update_cov ID error")
#             return  ## error
#
#         # Decay other incStat
#         self.incStats[not (inc)].processDecay(t)
#
#         # Decay residules
#         self.processDecay(t, inc)
#
#         # Compute and update residule
#         res = (v - self.incStats[inc].mean())
#         resid = (v - self.incStats[inc].mean()) * self.lastRes[not (inc)]
#         self.CF3 += resid
#         self.w3 += 1.
#         self.lastRes[inc] = res
#
#     def processDecay(self, t, micro_inc_indx):
#
#         factor = torch.pow(2, (- (self.incStats[micro_inc_indx].Lambda) * F.relu(t - self.lastTimestamp_cf3)))
#         self.CF3 *= factor
#         self.w3 *= factor
#         self.lastTimestamp_cf3 = t
#         self.lastRes[micro_inc_indx] *= factor
#         return factor
#
#     def cov(self):
#         return self.CF3 / self.w3
#
#     # Pearson corl. coef
#     def pcc(self):
#         ss = self.incStats[0].std() * self.incStats[1].std()
#         if ss != 0:
#             return self.cov() / ss
#         else:
#             return torch.tensor(0.).double()
#
#     # calculates and pulls all correlative stats AND 2D stats from both streams (incStat)
#     def get_stats2(self):
#         return torch.cat([self.incStats[0].radius([self.incStats[1]]).reshape(1, -1),
#                           self.incStats[0].magnitude([self.incStats[1]]).reshape(1, -1),
#                           self.cov().reshape(1, -1),
#                           self.pcc().reshape(1, -1)])
#
#
# class incStatDB:
#     # default_lambda: use this as the lambda for all streams. If not specified, then you must supply a Lambda with every query.
#     def __init__(self, limit=torch.tensor(float('inf'))):
#         self.HT = dict()
#         self.limit = limit
#
#     # Registers a new stream. init_time: init lastTimestamp of the incStat
#     def register(self, ID, Lambda, init_time, isTypeDiff=False):
#
#         key = ID + "_" + str(Lambda)
#         incS = self.HT.get(key)
#         if incS is None:
#             if len(self.HT) + 1 > self.limit:
#                 raise LookupError(
#                     'Adding Entry:\n' + key + '\nwould exceed incStatHT 1D limit of ' + str(
#                         self.limit) + '.\nObservation Rejected.')
#             incS = incStat(Lambda, ID, init_time, isTypeDiff)
#             self.HT[key] = incS
#         return incS
#
#     # Registers covariance tracking for two streams, registers missing streams
#     def register_cov(self, ID1, ID2, Lambda, init_time, isTypeDiff=False):
#
#         # Lookup both streams
#         incS1 = self.register(ID1, Lambda, init_time, isTypeDiff)
#         incS2 = self.register(ID2, Lambda, init_time, isTypeDiff)
#
#         # check for pre-exiting link
#         for cov in incS1.covs:
#             if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
#                 return cov
#
#         # Link incStats
#         inc_cov = incStat_cov(incS1, incS2, init_time)
#         incS1.covs.append(inc_cov)
#         incS2.covs.append(inc_cov)
#         return inc_cov
#
#     # Updates and then pulls current 1D stats from the given ID. Automatically registers previously unknown stream IDs
#     def update_get_1D_Stats(self, ID, t, v, Lambda=1, isTypeDiff=False):
#         incS = self.register(ID, Lambda, t, isTypeDiff)
#         incS.insert(v, t)
#         return incS.allstats_1D()
#
#     def update_get_2D_Stats(self, ID1, ID2, t1, v1, Lambda):
#         # retrieve/add cov tracker
#         inc_cov = self.register_cov(ID1, ID2, Lambda, t1)
#         # Update cov tracker
#         inc_cov.update_cov(ID1, v1, t1)
#         return inc_cov.get_stats2()
#
#     # Updates and then pulls current 1D and 2D stats from the given IDs. Automatically registers previously unknown stream IDs
#     def update_get_1D2D_Stats(self, ID1, ID2, t1, v1, Lambda=1):  # weight, mean, std
#         stats1d = self.update_get_1D_Stats(ID1, t1, v1, Lambda)
#         stats2d = self.update_get_2D_Stats(ID1, ID2, t1, v1, Lambda)
#         return torch.cat([stats1d, stats2d])