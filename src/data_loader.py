#!pip install volumentations-
#!pip install cv2
#!apt-get update
#!apt-get install ffmpeg libsm6 libxext6  -y

##################
#python libraries#
##################
import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.python.keras import (Input, Model)
##################
#  my libraries  #
##################
from volumentations import Compose
from volumentations import augmentations as ai
#from src.daTechniques import jitter, scaling, permutation, discriminative_guided_warp, time_warp, window_slice, spawner, rotation


def loadUncertainties(path):
    XTrainPor3D              = np.load(path + "train_POR3D.npz")
    XTrainRtp3D              = np.load(path + "train_RTP3D.npz")
    XTrainNtg3D              = np.load(path + "train_NTG3D.npz")
    XTrainPermi3D            = np.load(path + "train_PERMI3D.npz")
    XTrainfluidUncertainties = np.load(path + "train_fluidUncertainties.npz")

    XTestPor3D               = np.load(path + "test_POR3D.npz")
    XTestRtp3D               = np.load(path + "test_RTP3D.npz")
    XTestNtg3D               = np.load(path + "test_NTG3D.npz")
    XTestPermi3D             = np.load(path + "test_PERMI3D.npz")
    XTestfluidUncertainties  = np.load(path + "test_fluidUncertainties.npz")

    XValPor3D                = np.load(path + "validation_POR3D.npz")
    XValRtp3D                = np.load(path + "validation_RTP3D.npz")
    XValNtg3D                = np.load(path + "validation_NTG3D.npz")
    XValPermi3D              = np.load(path + "validation_PERMI3D.npz")
    XValfluidUncertainties   = np.load(path + "validation_fluidUncertainties.npz")

    XTrainPor3D               = XTrainPor3D.f.arr_0             
    XTrainRtp3D               = XTrainRtp3D.f.arr_0             
    XTrainNtg3D               = XTrainNtg3D.f.arr_0             
    XTrainPermi3D             = XTrainPermi3D.f.arr_0           
    XTrainfluidUncertainties  = XTrainfluidUncertainties.f.arr_0

    XTestPor3D                = XTestPor3D.f.arr_0              
    XTestRtp3D                = XTestRtp3D.f.arr_0              
    XTestNtg3D                = XTestNtg3D.f.arr_0              
    XTestPermi3D              = XTestPermi3D.f.arr_0             
    XTestfluidUncertainties   = XTestfluidUncertainties.f.arr_0 

    XValPor3D                 = XValPor3D.f.arr_0               
    XValRtp3D                 = XValRtp3D.f.arr_0               
    XValNtg3D                 = XValNtg3D.f.arr_0               
    XValPermi3D               = XValPermi3D.f.arr_0               
    XValfluidUncertainties    = XValfluidUncertainties.f.arr_0  

    
    return XTrainPor3D, XTrainRtp3D, XTrainNtg3D, XTrainPermi3D, XTrainfluidUncertainties, XTestPor3D, XTestRtp3D, XTestNtg3D, XTestPermi3D, XTestfluidUncertainties, XValPor3D, XValRtp3D, XValNtg3D, XValPermi3D, XValfluidUncertainties

def loadProductionANDHistory(path, String="", typeOfSet=None):
    if String == "Production":
        production = [x for x in os.listdir(path) if x.endswith(".npz") and x.split('_')[0] == typeOfSet]
        return sorted(production, key=lambda x: int(x.split('_')[1][:2]))
    else:
        history = [x for x in os.listdir(path) if x.endswith(".npz")]
        return sorted(history, key=lambda x: int(x[:2]))

def selectProductionCurve(listOfProductionCurves, dictionary):
    return sum([listOfProductionCurves[v[0]:v[1]] for k, v in dictionary.items()], [])

def loadUncertainties1D(path, reducer):
    XTrain              = np.load(path + reducer +'/Uncertainties/' + 'train_'+ reducer +'.npz')
    XTest               = np.load(path + reducer +'/Uncertainties/' + 'test_'+ reducer +'.npz')
    XVal                = np.load(path + reducer +'/Uncertainties/' + 'validation_'+ reducer +'.npz')
    XTrain               = XTrain.f.arr_0
    XTest                = XTest.f.arr_0 
    XVal                 = XVal.f.arr_0

    return XTrain, XTest, XVal