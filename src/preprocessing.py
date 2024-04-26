##################
#python libraries#
##################
import os
import random
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from functools import partial
from multiprocessing import cpu_count
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
##################
#  my libraries  #
##################
from utils import createPKLFile, clear_and_save

def random(s):
    """
    Set random seeds for numpy and tensorflow.
    """
    np.random.seed(s)
    tf.random.set_seed(s)  
            
def saveNormalizedOutput(pathtoOutput, pathToSaveOutput, dataType, randtrain, nSamplesTrain, typeOfOutput="/"):
    """
    Save normalized output data.
    """
    list_model = [x for x in os.listdir(pathtoOutput) if x.endswith(".npz")]
    list_model = sorted(list_model, key=lambda x: int(x[:2]))   
    
    pathToSaveOutput = pathToSaveOutput + typeOfOutput + dataType
    for y in list_model:
        output = np.load(pathtoOutput + y)
        output = output.f.arr_0
        if typeOfOutput=='ObjectiveFunction/':
            y_train = np.array([output[i] for i in randtrain])
            y_test  = np.array([output[i] for i, _ in enumerate(output) if i not in randtrain])
            y_validation = y_train[nSamplesTrain:]
            y_train = y_train[:nSamplesTrain]
            np.savez_compressed(pathToSaveOutput + "train_" + y.split('.')[0], y_train)
            np.savez_compressed(pathToSaveOutput + "test_" + y.split('.')[0], y_test)
            np.savez_compressed(pathToSaveOutput + "validation_" + y.split('.')[0], y_validation)
        else:
            np.savez_compressed(pathToSaveOutput + y.split('.')[0], output)

def getSplit(feature, randtrain, nSamplesTrain):
    """
    Split dataset into train, test, and validation sets.
    """
    train = np.array([feature[i] for i in randtrain])
    test = np.array([feature[i] for i, _ in enumerate(feature) if i not in randtrain])
    validation = train[nSamplesTrain:]
    train = train[:nSamplesTrain]    
    return train, test, validation

def splitDataset(pathToSaveOurOutput, randtrain, nSamplesTrain):
    """
    Split dataset into train, test, and validation sets and save as pickle file.
    """
    dictSplitData = {}
    splittrain = randtrain[:nSamplesTrain].tolist()
    dictSplitData['train'] = splittrain
    splitVal = randtrain[nSamplesTrain:].tolist()
    dictSplitData['validation'] = splitVal
    splitTest = [i for i in range(200) if i not in randtrain]
    dictSplitData['test'] = splitTest

    path = pathToSaveOurOutput + 'splitData'

    createPKLFile(path, dictSplitData)
    clear_and_save(path, dictSplitData)

def process_uncertainty(path_to_uncertainty, uncertainty, randtrain, n_samples_train, n_samples_test, n_samples_val, normalization_technique, path_to_save):
    """
    Process uncertainty data.
    """
    feature = np.load(path_to_uncertainty + uncertainty)
    feature = feature.f.arr_0
    if uncertainty != 'fluidUncertainties.npz':
        nsamples, timeStep, rows, cols, ch = feature.shape
        feature = np.reshape(feature, (nsamples, timeStep * ch * rows * cols))

    train, test, validation = getSplit(feature, randtrain, n_samples_train)

    train = normalization_technique.fit_transform(train)
    test = normalization_technique.transform(test)
    validation = normalization_technique.transform(validation)

    if uncertainty != 'fluidUncertainties.npz':
        train = np.reshape(train, (n_samples_train, timeStep, rows, cols, ch))
        test = np.reshape(test, (n_samples_test, timeStep, rows, cols, ch))
        validation = np.reshape(validation, (n_samples_val, timeStep, rows, cols, ch))

    path_train = path_to_save + 'Uncertainties/' + "train_" + uncertainty.split('.')[0]
    np.savez_compressed(path_train, train)

    path_test = path_to_save + 'Uncertainties/' + "test_" + uncertainty.split('.')[0]
    np.savez_compressed(path_test, test)

    path_val = path_to_save + 'Uncertainties/' + "validation_" + uncertainty.split('.')[0]
    np.savez_compressed(path_val, validation)
    
def preprocessing(pathToUncertainti, 
                  pathToProduction,
                  pathToHistory,
                  pathToSave,
                  randtrain,
                  normalizationTechnique, 
                  isSomeDays, nSamplesTrain, nSamplesVal, nSamplesTest):
    """
    Preprocess data.
    """
    listUncertainties = ['POR3D.npz', 'PERMI3D.npz', 
                         'NTG3D.npz', 'RTP3D.npz', 
                         'fluidUncertainties.npz']
        
    process_func = partial(process_uncertainty, pathToUncertainti, randtrain=randtrain,
                           n_samples_train=nSamplesTrain, n_samples_test=nSamplesTest, n_samples_val=nSamplesVal,
                           normalization_technique=normalizationTechnique,
                           path_to_save=pathToSave)

    with Pool(processes=cpu_count()) as pool:
        pool.map(process_func, listUncertainties)
          
    saveNormalizedOutput(pathToProduction + isSomeDays+'/', 
                         pathToSave, 
                         isSomeDays+'/', 
                         randtrain, nSamplesTrain,
                         typeOfOutput='ObjectiveFunction/')

    saveNormalizedOutput(pathToHistory + isSomeDays+'/', 
                         pathToSave,
                         isSomeDays+'/', 
                         randtrain, nSamplesTrain,
                         typeOfOutput='Observed/')           
    
def main(pathToUncertainties, 
         pathToObjectiveFunction,
         pathToObserved, 
         pathToSaveOutput,
         isSomeDays):
    """
    Normalize input uncertainties and output objective functions and observed data.
    
    Args:
    - path_to_uncertainties (str): Path to the folder containing uncertainty data.
    - path_to_objective_function (str): Path to the folder containing objective function data.
    - path_to_observed (str): Path to the folder containing observed data.
    - path_to_save_output (str): Path to save the normalized output.
    - is_some_days (str): Some specific information related to the data.
    - n_samples_train (int): Number of samples for training.
    - n_samples_val (int): Number of samples for validation.
    - n_samples_test (int): Number of samples for testing.
    """
    # seed    
    randtrain = np.arange(200)
    np.random.shuffle(randtrain)
    nSamplesTrain = 100
    nSamplesVal = 40
    nSamplesTest = 60
    randtrain = randtrain[:nSamplesTrain + nSamplesVal]
    
    # split train, test and validation set
    splitDataset(pathToSaveOutput, randtrain, nSamplesTrain)
    
    # normalization technique
    normalizationTechnique = QuantileTransformer(output_distribution= 'normal')
    
    preprocessing(
        pathToUncertainties,
        pathToObjectiveFunction,
        pathToObserved,
        pathToSaveOutput,
        randtrain,
        normalizationTechnique,
        isSomeDays, 
        nSamplesTrain, 
        nSamplesVal, 
        nSamplesTest
    )     

if __name__ == "__main__":
    main()