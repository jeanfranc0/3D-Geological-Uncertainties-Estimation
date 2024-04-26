##################
#python libraries#
##################
import os
import sys
import pickle
import shutil
import zipfile
import numpy as np
    
def deleteAllFilesAndFolder(path):
    """
    Deletes all files and folders at the specified path.

    Args:
        path (str): Path to the folder to be deleted.
    """
    shutil.rmtree(path)

def unZipFiles(pathToSave, path):
    """
    Unzips files from the specified path and saves them to the target path.

    Args:
        pathToSave (str): Path to save the unzipped files.
        path (str): Path to the zip file.
    """
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(pathToSave)

def deleteFilesfromFolder(folder, extention):
    """
    Deletes files with the specified extension from the specified folder.

    Args:
        folder (str): Path to the folder containing the files.
        extention (str): File extension to be deleted.
    """
    files = [x for x in os.listdir(folder) if x.endswith(extention)]
    for item in files:
        os.remove(os.path.join(folder,item))

def loadTXTFile(file):
    """
    Loads data from a text file.

    Args:
        file (str): Path to the text file.

    Returns:
        str: Contents of the text file.
    """
    with open(file) as f:
        lines = f.readlines()
    return lines[0]

def loadNpFile(path, y):
    """
    Loads data from a numpy file.

    Args:
        path (str): Path to the directory containing the numpy file.
        y (str): Name of the numpy file.

    Returns:
        np.ndarray: Loaded numpy array.
    """
    y = np.load(path + y)
    return y.f.arr_0
    
def clear_and_save(filename, newData):
    """
    Clears the content of the specified file and saves new data.

    Args:
        filename (str): Name of the file.
        newData: Data to be saved.
    """
    with open(filename+'.pkl', 'wb') as output:
        pickle.dump(newData, output)

def createPKLFile(filename, newData):
    """
    Creates and saves a pickle file with the specified data.

    Args:
        filename (str): Name of the file.
        newData: Data to be saved.
    """
    with open(filename+'.pkl', 'wb') as output:
        pickle.dump(newData, output)

def load_pkl_file(filename):
    """
    Loads data from a pickle file.

    Args:
        filename (str): Name of the pickle file.

    Returns:
        dict: Loaded dictionary.
    """
    with open(filename+'.pkl', "rb") as fp:
        dictionary = pickle.load(fp)
    return dictionary

def followingFile(filename, pozo):
    """
    Appends the specified content to the specified file.

    Args:
        filename (str): Name of the file.
        pozo (str): Content to be appended.
    """
    with open(filename, 'w') as f:
        f.write(pozo)
        f.write('\n')
        
def convertFoldertoZipFile(path, output_filename, dir_name):
    """
    Converts the specified folder to a zip file.

    Args:
        path (str): Path to the folder to be converted.
        output_filename (str): Name of the output zip file.
        dir_name (str): Name of the directory to be zipped.
    """
    shutil.make_archive(output_filename, 'zip', dir_name)