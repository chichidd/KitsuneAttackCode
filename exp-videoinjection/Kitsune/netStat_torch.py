import numpy as np

import AfterImage_torch as af

import torch


class netStat:
    # Datastructure for efficent network stat queries
    # HostLimit: no more that this many Host identifiers will be tracked
    # HostSimplexLimit: no more that this many outgoing channels from each host will be tracked (purged periodically)
    # Lambdas: a list of 'window sizes' (decay factors) to track for each stream. nan resolved to default [5,3,1,.1,.01]
    def __init__(self, Lambdas=None, HostLimit=255, HostSimplexLimit=1000):
        # Lambdas
        if Lambdas is None:
            self.Lambdas = torch.tensor([5., 3., 1., .1, .01])
        else:
            self.Lambdas = Lambdas

        # HT Limits
        self.HostLimit = HostLimit
        self.SessionLimit = HostSimplexLimit * self.HostLimit * self.HostLimit  # *2 since each dual creates 2 entries in memory
        self.MAC_HostLimit = self.HostLimit * 10

        # HTs
        self.HT_jit = af.incStatDB(limit=self.HostLimit * self.HostLimit, Lambdas=self.Lambdas)  # H-H Jitter Stats
        self.HT_MI = af.incStatDB(limit=self.MAC_HostLimit, Lambdas=self.Lambdas)  # MAC-IP relationships
        self.HT_H = af.incStatDB(limit=self.HostLimit, Lambdas=self.Lambdas)  # Source Host BW Stats
        self.HT_Hp = af.incStatDB(limit=self.SessionLimit, Lambdas=self.Lambdas)  # Source Host BW Stats

    def updateGetStats(self, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp, printNew=False):
        # no use for IPtype

        res = []

        res += self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize)

        res += self.HT_H.update_get_1D2D_Stats(srcIP, dstIP, timestamp, datagramSize)

        res += self.HT_jit.update_get_1D_Stats(srcIP + dstIP, timestamp, torch.tensor(0.) * datagramSize, isTypeDiff=True)


        if srcProtocol == 'arp':
            res += self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp, datagramSize)
        else:
            res += self.HT_Hp.update_get_1D2D_Stats(srcIP + srcProtocol, dstIP + dstProtocol, timestamp, datagramSize, printNew=printNew)

        return torch.stack(res)

    # def updateGetStats(self, IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp):
    #     # MAC.IP: Stats on src MAC-IP relationships
    #     MIstat = torch.zeros((3 * len(self.Lambdas), 1))
    #     for i in range(len(self.Lambdas)):
    #         MIstat[(i * 3):((i + 1) * 3)] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
    #                                                                        self.Lambdas[i])
    #
    #     # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
    #     HHstat = torch.zeros((7 * len(self.Lambdas), 1))
    #     for i in range(len(self.Lambdas)):
    #         HHstat[(i * 7):((i + 1) * 7)] = self.HT_H.update_get_1D2D_Stats(srcIP, dstIP, timestamp, datagramSize,
    #                                                                         self.Lambdas[i])
    #
    #     # Host-Host Jitter:
    #     HHstat_jit = torch.zeros((3 * len(self.Lambdas), 1))
    #     for i in range(len(self.Lambdas)):
    #         HHstat_jit[(i * 3):((i + 1) * 3)] = self.HT_jit.update_get_1D_Stats(srcIP + dstIP, timestamp, torch.tensor(
    #             0.).double() * datagramSize,
    #                                                                             self.Lambdas[i], isTypeDiff=True)
    #
    #     # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
    #     HpHpstat = torch.zeros((7 * len(self.Lambdas), 1))
    #     if srcProtocol == 'arp':
    #         for i in range(len(self.Lambdas)):
    #             HpHpstat[(i * 7):((i + 1) * 7)] = self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp,
    #                                                                                datagramSize, self.Lambdas[i])
    #     else:  # some other protocol (e.g. TCP/UDP)
    #         for i in range(len(self.Lambdas)):
    #             HpHpstat[(i * 7):((i + 1) * 7)] = self.HT_Hp.update_get_1D2D_Stats(srcIP + srcProtocol,
    #                                                                                dstIP + dstProtocol, timestamp,
    #                                                                                datagramSize, self.Lambdas[i])
    #
    #     # return MIstat, HHstat, HHstat_jit, HpHpstat
    #     return torch.cat([MIstat, HHstat, HHstat_jit, HpHpstat])  # concatenation of stats into one stat vector

