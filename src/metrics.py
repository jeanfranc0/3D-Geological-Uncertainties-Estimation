##################
#python libraries#
##################
import numpy as np

def rmse(y1, y2):
    y1, y2 = np.array(y1), np.array(y2)
    print('rmse:',np.sqrt(((y1-y2)**2).mean()))

def r2(y_true, y_pred):
    corr = np.corrcoef(y_true, y_pred)
    corr = corr[0,1]
    print('correlation:', corr)
    print('r2:', corr ** 2)
    
def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    print('mae:',np.mean(np.abs(y_true - predictions)))

def smape(y_true, predictions):
    numerador = np.abs(y_true - predictions)
    denominador =  (np.abs(y_true) + np.abs(predictions))
    return 100 * np.mean(numerador/denominador)