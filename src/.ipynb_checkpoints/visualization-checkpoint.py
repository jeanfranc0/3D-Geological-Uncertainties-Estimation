##################
#python libraries#
##################
import os
import sys
import math
import joblib
import sklearn
import numpy as np
import pandas as pd
from math import sqrt
import tensorflow as tf
import matplotlib.pyplot as plt
from  scipy.stats import linregress
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
##################
#  my libraries  #
##################
from src.metrics import smape
from utils import loadTXTFile, load_pkl_file, createPKLFile, followingFile, loadNpFile

def visualizationHistoryvsProxy(simulated, proxy, fileName, isSomeDays, pozo, typeOfSet, name, SMAPE =None, setSizeW=None):
    # plot simulated
    for impacts in simulated[:-1]:
        plt.plot(np.concatenate((impacts[:5], [np.nan], 
                                 impacts[5:10], [np.nan], 
                                 impacts[10:15], [np.nan],
                                 impacts[15:20], [np.nan],
                                 impacts[20:25], [np.nan],
                                 impacts[25:30], [np.nan],
                                 impacts[30:35], [np.nan])),'gray')
        plt.ylabel('Production', fontsize=16)
    
    plt.plot(np.concatenate((simulated[-1][:5], [np.nan], 
                             simulated[-1][5:10], [np.nan], 
                             simulated[-1][10:15], [np.nan],
                             simulated[-1][15:20], [np.nan],
                             simulated[-1][20:25], [np.nan],
                             simulated[-1][25:30], [np.nan],
                             simulated[-1][30:35], [np.nan])) , 'gray', label='simulated')
    
    #plot proxy
    for impacts in proxy[:-1]:
        plt.plot(np.concatenate((impacts[:5], [np.nan], 
                                 impacts[5:10], [np.nan], 
                                 impacts[10:15], [np.nan],
                                 impacts[15:20], [np.nan],
                                 impacts[20:25], [np.nan],
                                 impacts[25:30], [np.nan],
                                 impacts[30:35], [np.nan])),'limegreen')
    plt.plot(np.concatenate((proxy[-1][:5], [np.nan], 
                             proxy[-1][5:10], [np.nan], 
                             proxy[-1][10:15], [np.nan],
                             proxy[-1][15:20], [np.nan],
                             proxy[-1][20:25], [np.nan],
                             proxy[-1][25:30], [np.nan],
                             proxy[-1][30:35], [np.nan])) , 'limegreen', label='proxy')
    
    #plt.show()
    plt.title('Comparative results between Simulator and Proxy ', fontsize=15)
    custom_handles = [mpatches.Patch(color='gray', linewidth=1),
                      mpatches.Patch(color='limegreen', linewidth=1)]
                      #mpatches.Patch(color='none', linewidth=1)]
    
    if globlRange:
        new_x_ticks = ['$t_{506}$', '$t_{1014}$', '$t_{1360}$','$t_{1775}$', '$t_{2156}$']
        plt.xlabel('t', fontsize=16)
        plt.xticks(np.arange(len(new_x_ticks)), new_x_ticks, rotation=0, ha='right', fontsize=14, color='k')
    else:
        new_x_ticks = ['$t_{506}$', ' ', ' ',' ', '$t_{2156}$',''] * 7
        plt.xlabel('t', fontsize=16)
        plt.xticks(np.arange(len(new_x_ticks)), new_x_ticks, rotation=45, ha='right', fontsize=14, color='k')
    plt.yticks(rotation=0, ha='right', fontsize=14, color='k')
    
    # Define the range for the y-axis
    if setSizeW == 'G':
        plt.ylim(0, 150000000)
    elif setSizeW == 'W':
        plt.ylim(0, 1700000)
    elif setSizeW == 'B':
        plt.ylim(18000, 30000)
    else:
        plt.ylim(0, 2400000)
        
    plt.legend(handles=custom_handles, labels=['Simulated', 'Proxy'],fontsize=16)
    #plt.show()
    
    #followingFile(fileName+'following.txt', pozo)
    plt.savefig(fileName + isSomeDays + '/Plots/' + setSizeW+'/'+pozo + '/' + typeOfSet + '/'+ name + '.png')
    plt.show()
    plt.close()

def saveMyModel(model, fileName, isReused, pozo):
    lines = ['Readme', 'How to write text files in Python']
    with open(fileName + 'Models/' + isReused + '/' + pozo + '/readme.txt', 'w') as f:
        f.writelines(lines)
    #joblib.dump(model, fileName + 'Models/' + isReused + '/' + pozo + '/' + pozo + '.pkl')

def visualizationResults(y, fileName, pozo, typeOfSet, name, histo, someDays=None, setSizeW=None):
    colorChoosed = 'gray' if name == 'Simulated' else 'green'
    for impacts in y[: -1]:
        plt.plot(np.concatenate((impacts[:5], [np.nan], 
                                 impacts[5:10], [np.nan], 
                                 impacts[10:15], [np.nan],
                                 impacts[15:20], [np.nan],
                                 impacts[20:25], [np.nan],
                                 impacts[25:30], [np.nan],
                                 impacts[30:35], [np.nan])), color=colorChoosed)
        plt.ylabel('Production', fontsize=16)
    
    name = 'simulated' if name == 'Simulated' else 'proxy'
    plt.plot(np.concatenate((y[-1][:5], [np.nan], 
                                 y[-1][5:10], [np.nan], 
                                 y[-1][10:15], [np.nan],
                                 y[-1][15:20], [np.nan],
                                 y[-1][20:25], [np.nan],
                                 y[-1][25:30], [np.nan],
                                 y[-1][30:35], [np.nan])), 
             color=colorChoosed,
             label=name)
    plt.ylabel('Production', fontsize=16)
    plt.title(name, fontsize=20)
    plt.legend()
    if someDays == 'someDays':
        if globlRange:
            new_x_ticks = ['$t_{506}$', '$t_{1014}$', '$t_{1360}$','$t_{1775}$', '$t_{2156}$']
            plt.xlabel('t', fontsize=16)
            plt.xticks(np.arange(len(new_x_ticks)), new_x_ticks, rotation=45, ha='right', fontsize=14, color='k')
        else:
            new_x_ticks = ['$t_{506}$', ' ', ' ',' ', '$t_{2156}$', ''] * 7
            plt.xlabel('t', fontsize=6)
            plt.xticks(np.arange(len(new_x_ticks)), new_x_ticks, rotation=45, ha='right', fontsize=14, color='k')

    # Define the range for the x-axis
    plt.yticks(rotation=0, ha='right', fontsize=14, color='k')
    # Define the range for the y-axis
    if setSizeW == 'G':
        plt.ylim(0, 150000000)
    elif setSizeW == 'W':
        plt.ylim(0, 1700000)
    elif setSizeW == 'B':
        plt.ylim(17000, 30000)
    else:
        plt.ylim(0, 2400000)
    #fileName= '/tf/notebooks/jeanfranco/update6/Save/'
    plt.savefig(fileName + someDays + '/Plots/' + setSizeW+'/'+pozo + '/' + typeOfSet + '/'+ name + '.png')
    plt.show()
    plt.close()
    followingFile(fileName+'following.txt', pozo)
    
def plot_learning_curves(loss, val_loss):
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.close()
        
def combination(pozo, size, dictreductionSize):
    DA = ['DA', '']
    pozo = [p.split('.')[0][-3:] if i % 7 == 0 and p.split('.')[0][-3] == 'P' 
            else p.split('.')[0][-2:] for i, p in enumerate(pozo)]
    combinations = []
    for s in size:
        for p in pozo:
            for da in DA:
                combinations.append(p+dictreductionSize[str(s)]+da)
            
    return combinations

def saveSmape(pathToUploadProduction,
              someDaysTXT,
              ytrue, ypredict, 
              pozo, 
              fileName,
              isSomeDays,
              typeOfSet=None,
              setSizeW=None):   
    global globlRange 
    #falta agregar el numero de odelo notebooks/Notebooks/jeanfranco/preTrained/update5/preprocessing/splitData.txt
    if isSomeDays == 'someDays':
        days = [int(i) for i in loadTXTFile(someDaysTXT).split()] 
    else:
        days = os.path.join(os.path.dirname(pathToUploadProduction), 'model0001_W_m.rwo')
        days = pd.read_csv(days, sep ='\s+', skiprows = 6, engine='python')
        days = days.iloc[:, 0].tolist()
    if globlRange:
        days = range(5)
    else: 
        days = range(35)
        
    dictModels = load_pkl_file('/'.join(pathToUploadProduction.split('/')[:len(pathToUploadProduction.split('/')) - 3]) + '/normalization/splitData')

    Models = dictModels[typeOfSet]
    switch_dict = {
        'O': 'Oil',
        'G': 'Gas',
        'W': 'Water',
        'B': 'BHP'
    }
    if globlRange:
        if pozo[:3] == 'BHP':
            Choice = 'Analysis of the ' +switch_dict[setSizeW] + ' cumulative objective Function for the ' + pozo[3:][:-1]
        else:

            Choice = 'Analysis of the ' +  switch_dict[setSizeW] + ' cumulative objective Function for the ' + pozo[:-1]
    else:
        Choice = 'Analisys of the ' + pozo[:-1] + ' cumulative objective Function'
        
    for yTru, yPred, nroModel in zip(ytrue, ypredict, Models):
        plt.figure(figsize=(7, 7))
        plt.plot(days, yTru, 'gray', label="Simulated")#, 'g.')
        plt.plot(days, yPred, 'limegreen', label="Proxy")#, 'b.')
        plt.legend(loc="upper left")
        
        plt.ylabel('Production', fontsize=16)
        smp = smape(np.array(yTru), np.array(yPred))
        plt.title(Choice, fontsize=13)
        if globlRange:
            new_x_ticks = ['$t_{506}$', '$t_{1014}$', '$t_{1360}$','$t_{1775}$', '$t_{2156}$']
            plt.xlabel('t', fontsize=16)
            plt.xticks(np.arange(len(new_x_ticks)), new_x_ticks, rotation=0, ha='right', fontsize=14, color='k')
        else:
            new_x_ticks = ['$t_{506}$', ' ', ' ',' ', '$t_{2156}$'] * 7
            plt.xlabel('t', fontsize=6)
            plt.xticks(np.arange(len(new_x_ticks)), new_x_ticks, rotation=45, ha='right', fontsize=10, color='k')
        plt.yticks(rotation=0, ha='right', fontsize=14, color='k')
        if setSizeW == 'G':
            plt.ylim(0, 150000000)
        elif setSizeW == 'W':
            plt.ylim(0, 1700000)
        elif setSizeW == 'B':
            plt.ylim(0, 30000)
        else:
            plt.ylim(0, 2400000)
        custom_handles = [mpatches.Patch(color='gray', linewidth=1), 
                          mpatches.Patch(color='limegreen', linewidth=1), 
                          mpatches.Patch(color='none', linewidth=1)]
        plt.legend(handles=custom_handles, labels=['Simulated', 'Proxy'] + ['SMAPE: ' + str(round(smp,3))],
                   fontsize=16)
        #plt.show()
        plt.savefig(fileName + isSomeDays+ '/Plots/' + setSizeW+'/'+pozo + '/' + typeOfSet + '/model' + str(nroModel) + '.png')
        plt.close()
    
def computeSmape(pathToUploadProduction,
                 someDaysTXT,
                 yTest, yPredict,
                 pozo,
                 fileName,
                 isSomeDays, 
                 typeOfSet,
                 setSizeW=None):
    SmapeValues = []
    
    for index, y in enumerate(zip(yTest, yPredict)):
        #yTrueTest, yTest
        s = smape(y[0], y[1])
        SmapeValues.append(s)
    average = sum(SmapeValues)/len(SmapeValues)
    
    saveSmape(pathToUploadProduction,
              someDaysTXT,
              yTest, yPredict, 
              pozo, 
              fileName, 
              isSomeDays,
              typeOfSet,
              setSizeW=setSizeW)    
    return average, SmapeValues
    
def plotSmape(pathToUploadProduction,
              someDaysTXT,
              yTestRaw, y_test_prediction,
              yValRaw, y_validation_prediction,
              pozo, 
              fileName, 
              isSomeDays,
              setSizeW=None):
    # test
    averageTest, SmapeValuesTest = computeSmape(pathToUploadProduction,
                                                someDaysTXT,
                                                yTestRaw, y_test_prediction,
                                                pozo,
                                                fileName,
                                                isSomeDays, 
                                                'test',
                                                setSizeW=setSizeW)
    
    # validation
    averageValidation, SmapeValuesValidation = computeSmape(pathToUploadProduction,
                                                            someDaysTXT,
                                                            yValRaw, y_validation_prediction,
                                                            pozo,
                                                            fileName,
                                                            isSomeDays, 
                                                            'validation',
                                                            setSizeW=setSizeW)
    
    return averageTest, averageValidation

def plotCorrelationGraph(x, y, days, fileName, isReused, pozo, typeOfSet):
    xmin, xmax = min(x)- 1000, max(x + 1000)
    ymin, ymax = min(y)- 1000, max(y + 1000)
    plt.plot(x, y, 'o')
    
    start = (xmin, ymin)
    end = (xmax, ymax)
    
    plt.plot([start[0], end[0]], [start[1], end[1]], 'k--')
    plt.title('Correlation graph of day : ' + str(days), fontsize=13)
    plt.xlabel('simulated')
    plt.ylabel('proxy')
    plotCorrelationGraphwithDots(x, y, days, fileName, isReused, pozo, typeOfSet)
    
def plotCorrelationGraphwithDots(x, y, days, fileName, isReused, pozo, typeOfSet):
    xmin, xmax = min(x)- 1000, max(x + 1000)
    ymin, ymax = min(y)- 1000, max(y + 1000)
    
    plt.plot(x, y, 'ro')
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, y[i]], '--', color='red')
        plt.plot([0, x[i]], [y[i], y[i]], '--', color='blue')
        plt.plot(x[i], y[i], '.', color='white')

    start = (xmin, ymin)
    end = (xmax, ymax)
    plt.xlim(0, end[0])
    plt.ylim(0,end[1])
    plt.plot([start[0], end[0]], [start[1], end[1]], 'k--')
    plt.title('Correlation graph of day : ' + str(days), fontsize=13)
    plt.xlabel('simulated')
    plt.ylabel('proxy')
    plt.show()
    
def correlationGraph(simulated, proxy, someDaysTXT, fileName, isReused, pozo, typeOfSet, name, SMAPE =None):
    days = [int(i) for i in loadTXTFile(someDaysTXT).split()] 

    for i, j, d in zip(simulated.T, proxy.T, days):
        plotCorrelationGraph(i, j, d, fileName, isReused, pozo, typeOfSet)

def mape(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between the true and predicted values.
    
    Parameters:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.
    
    Returns:
    float: MAPE value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    np.seterr(divide='ignore', invalid='ignore')
import numpy as np

def rmsle(y_true, y_pred):
    """
    Calculates the Root Mean Squared Logarithmic Error (RMSLE) between the true and predicted values.
    
    Parameters:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.
    
    Returns:
    float: RMSLE value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred) 
    
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

def computeChiDistance(yTrue, yPred):
    concatenated_array = np.concatenate((yTrue, yPred), axis=1)
    FrequencyTable = concatenated_array/np.sum(concatenated_array)
    FrequencyTableSum = np.sum(concatenated_array, axis = 0, keepdims=True)/np.sum(concatenated_array)
    div = FrequencyTable / FrequencyTableSum

    yGroundTrue = div[:, 0]
    yPredicted = div[:, 1]

    denominador = (np.sum(concatenated_array, axis = 1, keepdims=True)/np.sum(concatenated_array)).flatten()

    return np.sqrt(np.sum(((yGroundTrue -  yPredicted) ** 2) / denominador))
    
def chi_distance(test_values, predicted_values):
    # https://link.springer.com/referenceworkentry/10.1007/978-0-387-32833-1_53#:~:text=The%20chi%2Dsquare%20distance%20incorporates,with%20more%20important%20sum%20package.
    chi_distance = []
    for yTrue, yPred in zip(test_values, predicted_values):
        yTrue = yTrue.reshape((-1, 1))
        yPred = yPred.reshape((-1, 1))
        chi = computeChiDistance(yTrue, yPred)
        chi_distance.append(chi)
    meanChi_distance = sum(chi_distance)/len(chi_distance)
    np.seterr(divide='ignore', invalid='ignore')
    return meanChi_distance

def newVisual(test_values, fileName, isSomeDays, predicted_values, SMAPE, TypeSet=None, setSizeW=None, pozo=None):
    # Calculate R2 score
    slope, intercept, r_value, p_value, std_err = linregress(test_values.flatten(), predicted_values.flatten())
    r2 = r2_score(test_values.flatten(), predicted_values.flatten())
    plt.scatter(predicted_values.flatten().tolist(), test_values.flatten().tolist(), color='blue', label='Proxy and Simulated')
    plt.xlabel('Proxy', fontsize=16)
    plt.ylabel('Simulated', fontsize=16)
    
    plt.title('Correlation between the Proxy and Simulator')
    if setSizeW == 'G':
        valuesminmax= [0, 150000000]
    elif setSizeW == 'W':
        valuesminmax= [0, 1700000]
    elif setSizeW == 'B':
        valuesminmax= [0, 30000]
    else:
        valuesminmax= [0, 2400000]
        
    plt.plot(valuesminmax, valuesminmax, color='red', linestyle='--', label='Ideal line')  
    plt.xticks(rotation=0, ha='right', fontsize=14, color='k')
    plt.yticks(rotation=0, ha='right', fontsize=14, color='k')
    
    # Display the legend with unique labels
    custom_handles = [mpatches.Patch(color='blue', linewidth=1), mpatches.Patch(color='red', linewidth=1), 
                      mpatches.Patch(color='none', linewidth=1), 
                      mpatches.Patch(color='none', linewidth=1),
                      mpatches.Patch(color='none', linewidth=1)]
    plt.legend(handles=custom_handles, labels=['Proxy and Simulated', 'ideal line', 
                                               'R: ' + str(round(r_value, 3)), 
                                               'R2: '+str(round(r2, 3)), 
                                               'SMAPE: ' + str(round(SMAPE, 3))],
               fontsize=12)
    
    if setSizeW == 'G':
        plt.xlim(0, 150000000)
        plt.ylim(0, 150000000)
    elif setSizeW == 'W':
        plt.xlim(0, 1700000)
        plt.ylim(0, 1700000)
    elif setSizeW == 'B':
        plt.xlim(18000, 30000)
        plt.ylim(18000, 30000)
    else:
        plt.xlim(0, 2400000)
        plt.ylim(0, 2400000)
    #plt.show()
    plt.savefig(fileName + isSomeDays + '/Plots/' + setSizeW+'/'+pozo + '/' + TypeSet + '/Correlation' + '.png')
    plt.show()
    plt.close()
    
globlRange = False

def computeAndSaveSmapeAndNQDS(pathToUploadProduction,
                               pathToUploadHistory,
                               someDaysTXT,
                               yTestRaw, y_test_prediction, 
                               yValRaw, y_validation_prediction,
                               fileName,
                               isSomeDays,
                               historyRaw):
    global globlRange    
    setSizeW = 'O'
    pozolist = ['oilO', 
                'waterW', 
                'gasG', 
                'bhpB']
    for i, namepozo in zip(range(0, 140, 35), pozolist):
        print(i, namepozo)
        print(namepozo[-1])
        if namepozo[-1] == 'G':
            setSizeW = 'G'
        if namepozo[-1] == 'B':
            setSizeW = 'B'
        if namepozo[-1] == 'W':
            setSizeW = 'W'
        history = loadNpFile(pathToUploadHistory + isSomeDays + '/', historyRaw)
        averageTest, averageValidation = plotSmape(pathToUploadProduction,
                                                   someDaysTXT,
                                                   yTestRaw[:, i:i+35], y_test_prediction[:, i:i+35],
                                                   yValRaw[:, i:i+35], y_validation_prediction[:, i:i+35],
                                                   namepozo, 
                                                   fileName, 
                                                   isSomeDays,
                                                   setSizeW=setSizeW)

        averageTest = round(averageTest, 3)
        averageValidation = round(averageValidation, 3)

        #newVisual(yTrainRaw, y_train_prediction)
        newVisual(yTestRaw[:, i:i+35], fileName, isSomeDays, y_test_prediction[:, i:i+35], averageTest, TypeSet = 'test',setSizeW=setSizeW,pozo=namepozo)
        #newVisual(yValRaw[:, i:i+35], fileName, isSomeDays, y_validation_prediction[:, i:i+35], averageValidation, TypeSet = 'validation',setSizeW=setSizeW,pozo=namepozo)

        #SMAPE for Test
        visualizationHistoryvsProxy(yTestRaw[:, i:i+35], 
                                    y_test_prediction[:, i:i+35],
                                    fileName,
                                    isSomeDays,
                                    namepozo,
                                    'test', 
                                    'testHistoryAndProxy', 
                                    SMAPE = averageTest,setSizeW=setSizeW)
        # SMAPE for validation
        #visualizationHistoryvsProxy(yValRaw[:, i:i+35], 
                                    # y_validation_prediction[:, i:i+35], 
                                    # fileName,
                                    # isSomeDays,
                                    # namepozo,
                                    # 'validation', 
                                    # 'valHistoryAndProxy', 
                                    # SMAPE = averageValidation,setSizeW=setSizeW)         

    return averageTest, averageValidation
    



