##################
#python libraries#
##################
import os
import sys
import pickle
import shutil
import zipfile
import numpy as np
p = os.path.abspath('../..')
if p not in sys.path:
    sys.path.append(p)
    
def deleteAllFilesAndFolder(path):
    shutil.rmtree(path)

def unZipFiles(pathToSave, path):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(pathToSave)

def deleteFilesfromFolder(folder, extention):
    #folder = "/tf/notebooks/Notebooks/jeanfranco/preTrained/update5/input/Production/"
    #extention = '.npz'
    files = [x for x in os.listdir(folder) if x.endswith(extention)]
    for item in files:
        os.remove(os.path.join(folder,item))

def loadTXTFile(file):
    with open(file) as f:
        lines = f.readlines()
    return lines[0]

def loadNpFile(path, y):
    y = np.load(path + y)
    return y.f.arr_0
    
def clear_and_save(filename, newData):
    with open(filename+'.pkl', 'wb') as output:
        # Pickle dictionary using protocol 0.
        pickle.dump(newData, output)

def createPKLFile(filename, newData):
    # to save the .pkl file
    print(filename, newData)
    with open(filename+'.pkl', 'wb') as output:
        # Pickle dictionary using protocol 0.
        pickle.dump(newData, output)

def load_pkl_file(filename):
    # load data from pkl file
    with open(filename+'.pkl', "rb") as fp:
        dictionary = pickle.load(fp)
    return dictionary

def followingFile(filename, pozo):
    with open(filename, 'w') as f:
        f.write(pozo)
        f.write('\n')
        
def convertFoldertoZipFile(path, output_filename, dir_name):
    #output_filename = '/tf/notebooks/jeanfranco/update5/Results/'
    #dir_name = destino
    shutil.make_archive(output_filename, 'zip', dir_name)