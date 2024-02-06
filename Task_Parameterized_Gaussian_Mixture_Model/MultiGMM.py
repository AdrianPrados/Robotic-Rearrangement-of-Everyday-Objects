import numpy as np
from .sClass import s
from .pClass import p
from matplotlib import pyplot as plt
from .TPGMM_GMR import TPGMM_GMR
from copy import deepcopy
import time
from scipy.stats import multivariate_normal
from scipy.integrate import nquad
import scipy
from collections import OrderedDict
from scipy.stats import zscore
import os

#directorio = '/home/hummer/Escritorio/Adam_Bot/generative_models/Task_Parameterized_Gaussian_Mixture_Model/carpeta/'
#directorio = '/home/hummer/Escritorio/Adam_Bot/generative_models/Task_Parameterized_Gaussian_Mixture_Model/datosDch/'

#* Estimation of the KL divergence ------------------------------------------------------------------------------------ #
def kl_mvn(to, fr):
    """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices"""
    m_to, S_to = to
    m_fr, S_fr = fr 
    d = m_fr - m_to
    c, lower = scipy.linalg.cho_factor(S_fr)
    def solve(B):
        return scipy.linalg.cho_solve((c, lower), B)
    def logdet(S):
        return np.linalg.slogdet(S)[1]
    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)
    return (term1 + term2 + term3 - len(d))/2.

def scale_value(original_value, min_original, max_original, min_new, max_new):
    return min_new + ((original_value - min_original) / (max_original - min_original)) * (max_new - min_new)

def normalize_values(values):
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    normalized_values = [(x - min_val) / range_val for x in values]
    return normalized_values


def main(C_img,C_final,DX_TP,DY_TP,DX_TPF,DY_TPF,direction,distance):
    C_img_TP=[0,0]
    C_img_TP0=[0,0]
    C_final_TP=[0,0]
    C_final_TP0=[0,0]
    #* Selection of the data to teach the TPGMM
    if direction == True and distance > 320: #* Left --> Rigth
        print("Left to Rigth")
        directorio = '/home/hummer/Escritorio/Adam_Bot/generative_models/Task_Parameterized_Gaussian_Mixture_Model/datosIzq/'
        samples = 25
        data = 200
        limitData = 3
    elif direction == False and distance > 320:
        print("Rigth to Left")
        directorio = '/home/hummer/Escritorio/Adam_Bot/generative_models/Task_Parameterized_Gaussian_Mixture_Model/datosDch/'
        samples = 25
        data = 200
        limitData = 3
    elif direction == True and distance <= 320:
        print("Short Left to Rigth")
        directorio = '/home/hummer/Escritorio/Adam_Bot/generative_models/Task_Parameterized_Gaussian_Mixture_Model/datosIzq75/'
        samples = 20
        data = 75
        limitData = 2
    elif direction == False and distance <= 320:
        print("Short Rigth to Left")
        directorio = '/home/hummer/Escritorio/Adam_Bot/generative_models/Task_Parameterized_Gaussian_Mixture_Model/datosDch75/'
        samples = 20
        data = 75
        limitData = 2
    
    #* Initialization of parameters and properties------------------------------------------------------------------------- #
    nbSamples = samples #Number of data
    nbVar = 3
    nbFrames = 2
    nbStates = 12 #Number of Gaussians
    nbData = data

    #* Preparing the samples----------------------------------------------------------------------------------------------- #
    slist = []
    AllFr=list(range(nbSamples))
    # En Slist gaurdamos los datos y los aprámetros: slist[n].Data serán losd atos y slist[n].p serán los aprámetros
    #Si queremos crear nuevos frames, tenemos que tener el tempB con el numero de data (por ejemplo 100) todo el rato los mismo valores
    for i in range(nbSamples):
        pmat = np.empty(shape=(nbFrames, nbData), dtype=object)
        tempData = np.loadtxt(directorio + 'sample' + str(i + 1) + '_Data.txt', delimiter=',',encoding='utf-8-sig')
        #print("Len tempData: {}".format(tempData0[0]))
        for j in range(nbFrames):
            tempA = np.loadtxt(directorio + 'sample' + str(i + 1) + '_frame' + str(j + 1) + '_A.txt', delimiter=',',encoding='utf-8-sig')
            tempB = np.loadtxt(directorio + 'sample' + str(i + 1) + '_frame' + str(j + 1) + '_b.txt', delimiter=',',encoding='utf-8-sig')
            for k in range(nbData):
                pmat[j, k] = p(tempA[:, 3*k : 3*k + 3], tempB[:, k].reshape(len(tempB[:, k]), 1),
                            np.linalg.inv(tempA[:, 3*k : 3*k + 3]), nbStates)
        slist.append(s(pmat, tempData, tempData.shape[1], nbStates))

    #* Creating instance of TPGMM_GMR-------------------------------------------------------------------------------------- #
    TPGMMGMR = TPGMM_GMR(nbStates, nbFrames, nbVar)
    print("Funciono hasta aqui")
    # Learning the model-------------------------------------------------------------------------------------------------- #
    TPGMMGMR.fit(slist)

    # Reproduction for parameters used in demonstration------------------------------------------------------------------- #
    rdemolist = []
    mus_init=[]
    sigmas_init=[]
    mus=[] #? Los valores se gaurdan en [n][0][j][i] donde n es el numero de demos y i el nuemro de gaussianas y j es la columna
    sigmas=[] #? los valores e gaurdan en [n][0][j][e][i] n y i son iguales; j es 0 o 1 que es la columna; e es entr 0 y 1 que es el primer o el segundo valor de la columna
    for n in range(nbSamples):
        rdemolist.append(TPGMMGMR.reproduce(slist[n].p, slist[n].Data[1:3,0]))
    #print("Datos dentro de rdemolist: {}".format(rdemolist[0].Mu[0]))
    
    #* Plotting------------------------------------------------------------------------------------------------------------ #
    xaxis = 1
    yaxis = 2
    xlim = [-1.2, 1.2]
    ylim = [-1.2, 1.2]

    C_img_TP[0] = scale_value(C_img[0],0,640,xlim[0],xlim[1])
    C_img_TP0[1] = 480 - C_img[1]
    C_img_TP[1] = scale_value(C_img_TP0[1],0,480,ylim[0],ylim[1])
    C_final_TP[0] = scale_value(C_final[0],0,640,xlim[0],xlim[1])
    C_final_TP0[1] = 480 - C_final[1]
    C_final_TP[1] = scale_value(C_final_TP0[1],0,480,ylim[0],ylim[1])
    

    print("C_img_TP: {}".format(C_img_TP))
    print("C_final_TP: {}".format(C_final_TP))

    #* Reproduction with just one new point ------------------------------------------------------------------------------ #
    rnewlist = []
    nSolutions= 1
    for n in range(nSolutions):
        newP = deepcopy(slist[1].p) #Copy the structure of an sClass
        for m in range(1, nbFrames): 
            for k in range(nbData): # It is executed nbData times
                newP[m, k].b = np.array([[0.],[C_final_TP[0]],[C_final_TP[1]]])
                newP[m, k].A = np.array([[1, 0.0, 0.0],[0.0,DY_TPF,DX_TPF],[0.0,DX_TPF,DY_TPF]])#np.array([[1, 0.0, 0.0],[0.0,0.12,-0.15],[0.0,-0.15,-0.12]])
                newP[m, k].invA = np.linalg.inv(newP[m, k].A)
                newP[m, k].Mu = np.zeros(shape=(len(newP[m, k].b ),nbStates))
                newP[m, k].Sigma = np.zeros(shape=(len(newP[m, k].b ),len(newP[m, k].b ),nbStates))

        for h in range(nbData):
            newP[0,h].b = np.array([[0.],[C_img_TP[0]],[C_img_TP[1]]]) #Modified the position of the initial points
            newP[0,h].A = np.array([[1, 0.0, 0.0],[0.0,DY_TP,DX_TP],[0.0,DX_TP,DY_TP]]) #Modified the orientation of the initial point
            """ newP[m, k].Mu = np.zeros(shape=(len(newP[0, k].b ),nbStates))
            newP[m, k].Sigma = np.zeros(shape=(len(newP[0, k].b ),len(newP[0, k].b ),nbStates)) """
        newPoint = np.array([C_img_TP[0], C_img_TP[1]])
        rnewlist.append(TPGMMGMR.reproduce(newP, newPoint))
    

    #* Demos--------------------------------------------------------------------------------------------------------------- #
    fig = plt.figure()
    ax1 = fig.add_subplot(141)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
    plt.title('Demonstrations')
    for n in range(nbSamples):
        for m in range(nbFrames):
            ax1.plot([slist[n].p[m,0].b[xaxis,0], slist[n].p[m,0].b[xaxis,0] + slist[n].p[m,0].A[xaxis,yaxis]], [slist[n].p[m,0].b[yaxis,0], slist[n].p[m,0].b[yaxis,0] + slist[n].p[m,0].A[yaxis,yaxis]], lw = 7, color = [0,1,m])
            ax1.plot(slist[n].p[m,0].b[xaxis,0], slist[n].p[m,0].b[yaxis,0], ms = 30, marker = '.', color = [0,1,m])
        ax1.plot(slist[n].Data[xaxis,0], slist[n].Data[yaxis,0], marker = '.', ms = 15)
        ax1.plot(slist[n].Data[xaxis,:], slist[n].Data[yaxis,:])
    #* Reproductions with training parameters------------------------------------------------------------------------------ #
    ax2 = fig.add_subplot(142)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
    plt.title('Reproductions with same task parameters')
    for n in range(nbSamples):
        listMu,listSig,_ = TPGMMGMR.plotReproduction(rdemolist[n], xaxis, yaxis, ax2, showGaussians=False)
        mus.append(listMu)
        sigmas.append(listSig)
    #* Reproductions with new parameters----------------------------------------------------------------------------------- #
    ax3 = fig.add_subplot(143)
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
    plt.title('Reproduction with generated task parameters')
    for n in range(nSolutions):
        listMu,listSig,_ = TPGMMGMR.plotReproduction(rnewlist[n], xaxis, yaxis, ax3,C_fin=C_final_TP, showGaussians=False)
        mus_init.append(listMu)
        sigmas_init.append(listSig)

    inicio = time.time()
    #? Generation of Gaussian comaprision
    IrrF = []
    cont=0
    Kl_mean = []
    Kl_val =0
    for h in range(nSolutions):
        for e in range (nbSamples): 
            #print("KL Calculo: {}".format(e))
            cont=0
            for i in range(nbStates):
                #print("Estado: {}".format(i))
                mean1 = np.concatenate(np.array([mus_init[h][0][0][i],mus_init[h][0][1][i]]))# El tercero 0 indica x si es 0 ,y si es 1
                cov1 = np.array([np.concatenate([sigmas_init[h][0][0][0][i], sigmas_init[h][0][0][1][i]]), np.concatenate([sigmas_init[h][0][1][0][i], sigmas_init[h][0][1][1][i]])]) # Matriz de covarianza de la primera gaussiana

                mean2 = np.concatenate(np.array([mus[e][0][0][i],mus[e][0][1][i]]))  # Media de la segunda gaussiana
                cov2 = np.array([np.concatenate([sigmas[e][0][0][0][i], sigmas[e][0][1][0][i]]), np.concatenate([sigmas[e][0][1][0][i], sigmas[e][0][1][1][i]])])  # Matriz de covarianza de la segunda gaussiana
                #kl_divergence_value = kl_divergence(gaussian1, gaussian2, show=False)
                try:
                    kl_divergence_value = kl_mvn([mean1, cov1],[mean2,cov2])
                    #print("KL Divergence:", kl_divergence_value)
                    Kl_val =Kl_val+ kl_divergence_value
                except:
                    cont = cont +1
                    #print("Excepcion matematica, sin divergencia de KL por lo tanto se descarta")
                    Kl_val = Kl_val+ 2000
                    """ if cont >=2:
                        IrrF.append(e) """
                    continue
            
            #print("++++++++++++")
            Kl_mean.append(abs(Kl_val))
            Kl_val = 0
    fin = time.time()
    #print(fin-inicio)
    print("Valores de Kl_mean: {}".format(Kl_mean))
    Kl_mean_Normalized = normalize_values(Kl_mean)
    print("Valores de Kl_mean normalizados: {}".format(Kl_mean_Normalized))
    time.sleep(2)
    """ dos_valores_mas_altos = sorted(Kl_mean, reverse=True)[:3]
    print(dos_valores_mas_altos) """
    for a in range(len(Kl_mean)):
        if Kl_mean_Normalized[a] > 0.8: #Threshold to select the correct data
            IrrF.append(a)
        else:
            #if Kl_mean[a] == dos_valores_mas_altos[1] or Kl_mean[a] == dos_valores_mas_altos[2]:
            if Kl_mean[a] in sorted(Kl_mean, reverse=True)[:limitData]:
                print(a)
                IrrF.append(a)


    #? Eliminate the irrelevanyt frames
    #print("Frames irrelevantes: {}".format(IrrF))
    IrrF = list(OrderedDict.fromkeys(IrrF)) #Eliminate duplicated values
    print("Frames irrelevantes: {}".format(IrrF))
    RelFr = list(set(AllFr).symmetric_difference(set(IrrF)))

    print("Frames relevantes: {}".format(RelFr))

    slistRelevant = []
    #Repite the proccess only with the relevant Frames
    for i in RelFr:
        pmat = np.empty(shape=(nbFrames, nbData), dtype=object)
        tempData = np.loadtxt(directorio + 'sample' + str(i + 1) + '_Data.txt', delimiter=',',encoding='utf-8-sig')
        for j in range(nbFrames):
            tempA = np.loadtxt(directorio + 'sample' + str(i + 1) + '_frame' + str(j + 1) + '_A.txt', delimiter=',',encoding='utf-8-sig')
            tempB = np.loadtxt(directorio +'sample' + str(i + 1) + '_frame' + str(j + 1) + '_b.txt', delimiter=',',encoding='utf-8-sig')
            for k in range(nbData):
                #print(tempA[:, 3*k : 3*k + 3])
                pmat[j, k] = p(tempA[:, 3*k : 3*k + 3], tempB[:, k].reshape(len(tempB[:, k]), 1),
                            np.linalg.inv(tempA[:, 3*k : 3*k + 3]), nbStates)
            
        slistRelevant.append(s(pmat, tempData, tempData.shape[1], nbStates))
    RelSamples = len(slistRelevant)
    #? Regenerate the model only with the relevant frames
    TPGMMGMR2 = TPGMM_GMR(nbStates, nbFrames, nbVar)
    TPGMMGMR2.fit(slistRelevant)

    for n in range(RelSamples):
        rdemolist.append(TPGMMGMR2.reproduce(slistRelevant[n].p, slistRelevant[n].Data[1:3,0]))

    rnewlist = []
    nSolutions= 1
    for n in range(nSolutions):
        newP = deepcopy(slistRelevant[n].p) #Copy the structure of an sClass
        for m in range(1, nbFrames): 
            for k in range(nbData): # It is executed nbData times
                newP[m, k].b = np.array([[0.],[C_final_TP[0]],[C_final_TP[1]]])
                newP[m, k].A = np.array([[1, 0.0, 0.0],[0.0,DY_TPF,DX_TPF],[0.0,DX_TPF,DY_TPF]]) #np.array([[1, 0.0, 0.0],[0.0,0.12,-0.15],[0.0,-0.15,-0.12]])
                newP[m, k].invA = np.linalg.inv(newP[m, k].A)
                newP[m, k].Mu = np.zeros(shape=(len(newP[m, k].b ),nbStates))
                newP[m, k].Sigma = np.zeros(shape=(len(newP[m, k].b ),len(newP[m, k].b ),nbStates))

        for h in range(nbData):
            newP[0,h].b = np.array([[0.],[C_img_TP[0]],[C_img_TP[1]]]) #Modified the position of the initial points
            newP[0,h].A = np.array([[1, 0.0, 0.0],[0.0,DY_TP,DX_TP],[0.0,DX_TP,DY_TP]]) #Modified the orientation of the initial point
        newPoint = np.array([C_img_TP[0], C_img_TP[1]])
        rnewlist.append(TPGMMGMR2.reproduce(newP, newPoint))
    ax4 = fig.add_subplot(144)
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
    plt.title('Reproduction with just relevant frames')
    for n in range(nSolutions):
        listMu,listSig,rData = TPGMMGMR2.plotReproduction(rnewlist[n], xaxis, yaxis, ax4,C_fin=C_final_TP, showGaussians=False)
    #print("DATOS: {}".format(rData))
    plt.show()
    return rData
    
if __name__ == "__main__":
    main(C_img,C_final,DX_TP,DY_TP,DX_TPF,DY_TPF,distance)