from .modelClass import model
from .init_proposedPGMM_timeBased import init_proposedPGMM_timeBased
from .EM_tensorGMM import EM_tensorGMM
from .reproduction_DSGMR import reproduction_DSGMR
from .plotGMM import plotGMM
import numpy as np
import time

class TPGMM_GMR(object):
    def __init__(self, nbStates, nbFrames, nbVar):
        self.model = model(nbStates, nbFrames, nbVar, None, None, None, None, None)
        

    def fit(self, s):
        self.s = s
        self.model = init_proposedPGMM_timeBased(s, self.model)
        self.model = EM_tensorGMM(s, self.model)

    def reproduce(self, p, currentPosition):
        return reproduction_DSGMR(self.s[0].Data[0,:], self.model, p, currentPosition)

    def plotReproduction(self, r, xaxis, yaxis, ax,C_fin = None, showGaussians = True, lw = 7):
        list_mu = []
        list_sigma=[]
        if C_fin is not None:
            r.Data[xaxis,-1]=C_fin[0]
            r.Data[yaxis,-1] = C_fin[1]
        #print(r.p.shape[0])
        for m in range(r.p.shape[0]):
            ax.plot([r.p[m, 0].b[xaxis, 0], r.p[m, 0].b[xaxis, 0] + r.p[m, 0].A[xaxis, yaxis]],
                    [r.p[m, 0].b[yaxis, 0], r.p[m, 0].b[yaxis, 0] + r.p[m, 0].A[yaxis, yaxis]],
                    lw=lw, color=[0, 1, m])
            ax.plot(r.p[m, 0].b[xaxis, 0], r.p[m, 0].b[yaxis, 0], ms=30, marker='.', color=[0, 1, m])
        ax.plot(r.Data[xaxis, 0], r.Data[yaxis, 0], marker='.', ms=15)
        ax.plot(r.Data[xaxis, :], r.Data[yaxis, :])
        ax.plot(r.Data[xaxis, -1], r.Data[yaxis, -1], marker='*', ms=15)
        if showGaussians:
            plotGMM(r.Mu[np.ix_([xaxis, yaxis], range(r.Mu.shape[1]), [0])],
                    r.Sigma[np.ix_([xaxis, yaxis], [xaxis, yaxis], range(r.Mu.shape[1]), [0])], [0.5, 0.5, 0.5], 1, ax)
        #print("Mu: {}".format(r.Mu[np.ix_([xaxis, yaxis], range(r.Mu.shape[1]), [0])]))
        #print("Sigma: {}".format(r.Sigma[np.ix_([xaxis, yaxis], [xaxis, yaxis], range(r.Mu.shape[1]), [0])]))
        #* Return the data from each Gaussian that generates the path
        list_mu.append(r.Mu[np.ix_([xaxis, yaxis], range(r.Mu.shape[1]), [0])])
        list_sigma.append(r.Sigma[np.ix_([xaxis, yaxis], [xaxis, yaxis], range(r.Mu.shape[1]), [0])])
        return list_mu,list_sigma,r.Data

    def getReproductionMatrix(self, r):
        return r.Data
