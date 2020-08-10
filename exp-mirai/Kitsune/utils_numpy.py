
import AfterImage_numpy as aft

def relu(x):
    if x>0:
        return x
    return 0

def incstatGetStats(incstat, t, v):
    '''
    auxillary function of getStats
    '''
    if incstat.isTypeDiff:
        v = relu(t - incstat.lastTimestamp)
    factor = math.pow(2, (-incstat.Lambda * relu(t - incstat.lastTimestamp)))
    CF1 = incstat.CF1 * factor + v
    CF2 = incstat.CF2 * factor + v ** 2
    w = incstat.w * factor + 1.
    cur_mean = CF1 / w
    cur_var = torch.abs(CF2 / w - cur_mean ** 2)
    return torch.stack([w, cur_mean, cur_var]).T.reshape(-1)


def incstatCovGetStats(incstat1, incstat2, t, v):
    '''
    auxillary function of getStats
    '''
    # update incstat ID1
    factor = torch.pow(2, (-incstat1.Lambda * F.relu(t - incstat1.lastTimestamp)))
    CF11 = incstat1.CF1 * factor + v
    CF21 = incstat1.CF2 * factor + v ** 2
    w1 = incstat1.w * factor + 1.
    cur_mean1 = CF11 / w1
    cur_var1 = torch.abs(CF21 / w1 - cur_mean1 ** 2)
    cur_std1 = torch.sqrt(cur_var1 + 1e-8)

    # get incstat_cov
    incstat_cov = None
    for cov in incstat1.covs:
        if cov.incStats[0].ID == incstat2.ID or cov.incStats[1].ID == incstat2.ID:
            incstat_cov = cov
            break
    exist_incstat_cov = True
    if incstat_cov is None:
        exist_incstat_cov = False
        incstat_cov = aft.incStat_cov(incstat1, incstat2, t)
        incstat1.covs.append(incstat_cov)
        incstat2.covs.append(incstat_cov)

    # update cov, inc is index of incstat1
    if incstat1.ID == incstat_cov.incStats[0].ID:
        inc = 0
    else:
        inc = 1

    # Decay other incStat

    factor2 = torch.pow(2, (-incstat2.Lambda * F.relu(t - incstat2.lastTimestamp)))
    CF12_decay = incstat2.CF1 * factor2
    CF22_decay = incstat2.CF2 * factor2
    w2_decay = incstat2.w * factor2

    # Decay residules
    # self.processDecay(t, inc)
    factor_inc = torch.pow(2, (-(incstat1.Lambda) * F.relu(t - incstat_cov.lastTimestamp_cf3)))
    CF3 = incstat_cov.CF3 * factor_inc
    w3 = incstat_cov.w3 * factor_inc

    # Compute and update residule
    res = (v - cur_mean1)

    resid = (v - cur_mean1) * incstat_cov.lastRes[not (inc)]
    if exist_incstat_cov:
        CF3 += resid * 2
        w3 += 2.
    else:
        # new incstat_cov, won't be updated during insert operation of incstat1
        CF3 += resid
        w3 += 1.

    # compute stats2
    if len(incstat2.cur_mean) == 1:
        incstat2_mean = CF12_decay / w2_decay
    else:
        incstat2_mean = incstat2.cur_mean

    if len(incstat2.cur_var) == 1:
        incstat2_var = torch.abs(CF22_decay / w2_decay - incstat2_mean ** 2)
    else:
        incstat2_var = incstat2.cur_var

    if len(incstat2.cur_std) == 1:
        incstat2_std = torch.sqrt(incstat2_var + 1e-8)
    else:
        incstat2_std = incstat2.cur_std

    radius = cur_var1 + incstat2_var#torch.sqrt(cur_var1 + incstat2_var + 1e-8)
    magnitude = cur_mean1 ** 2 + incstat2_mean ** 2 #torch.sqrt(cur_mean1 ** 2 + incstat2_mean ** 2 + 1e-8)
    cov = CF3 / w3

    # ss = cur_std1 * incstat2_std
    # pcc[pcc == float('inf')] = 0.
    # pcc[pcc == -float('inf')] = 0.
    # pcc[pcc != pcc] = 0.

    # pcc = (cov ** 2) / (cur_var1 * incstat2_var)  # cov / ss
    # for i in range(len(cov)):
    #     tmp = cur_var1[i]*incstat2_var[i]
    #     if tmp != 0:
    #         pcc[i] /= tmp
    #     else:
    #         pcc[i] *= 0.

    pcc = torch.sign(cov) * cov ** 2
    div = cur_var1 * incstat2_var
    pcc[div != 0] = pcc[div != 0] / div[div != 0]
    pcc[div == 0] *= 0


    return torch.stack([w1, cur_mean1, cur_var1, radius, magnitude, cov, pcc]).T.reshape(-1)


def getStats(nstat, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp):
    '''
    get next vector given <nstat> and next package without update of <nstat>
    '''
    res = []
    res += incstatGetStats(nstat.HT_MI.HT[srcMAC + srcIP], timestamp, datagramSize)

    res += incstatCovGetStats(nstat.HT_H.HT[srcIP], nstat.HT_H.HT[dstIP], timestamp, datagramSize)
    res += incstatGetStats(nstat.HT_jit.HT[srcIP + dstIP], timestamp, torch.tensor(0.) * datagramSize)
    if srcProtocol == 'arp':
        res += incstatCovGetStats(nstat.HT_Hp.HT[srcMAC], nstat.HT_Hp.HT[dstMAC], timestamp, datagramSize)
    else:
        res += incstatCovGetStats(nstat.HT_Hp.HT[srcIP + srcProtocol], nstat.HT_Hp.HT[dstIP + dstProtocol], timestamp,
                                  datagramSize)

    return torch.stack(res).reshape(1, -1)

def printIncstat(incstat):
    '''
    Print an incstat object
    '''
    print("ID: ", incstat.ID)
    print("CF1: ", incstat.CF1)
    print("CF2: ", incstat.CF2)
    print("w: ", incstat.w)
    print("lastts: ", incstat.lastTimestamp)
    print("mean: ", incstat.cur_mean)
    print("var: ", incstat.cur_var)
    print("std: ", incstat.cur_std)