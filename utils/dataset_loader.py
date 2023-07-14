import os
import pandas as pd
import random

def assign_dataset():
    # List of dataset filenames
    dataset_files = ['D1_DATASET.xlsx',
                     'D2_DATASET.xlsx',
                     'D3_DATASET.xlsx',
                     'D4_DATASET.xlsx',
                     'D5_DATASET.xlsx',
                     'D6_DATASET.xlsx',
                     'D7_DATASET.xlsx',
                     'D8_DATASET.xlsx',
                     'D9_DATASET.xlsx',
                     'D10_DATASET.xlsx',
                     'D11_DATASET.xlsx',
                     'D12_DATASET.xlsx',
                     'D13_DATASET.xlsx',
                     'D14_DATASET.xlsx',
                     'D15_DATASET.xlsx',
                     'D16_DATASET.xlsx',
                     'D17_DATASET.xlsx',
                     'D18_DATASET.xlsx',
                     'D19_DATASET.xlsx',
                     'D20_DATASET.xlsx',
                     'D21_DATASET.xlsx',
                     'D22_DATASET.xlsx',
                     'D23_DATASET.xlsx',
                     'D24_DATASET.xlsx',
                     'D25_DATASET.xlsx',
                     'D26_DATASET.xlsx',
                     'D27_DATASET.xlsx',
                     'D28_DATASET.xlsx',
                     'D29_DATASET.xlsx',
                     'D30_DATASET.xlsx']

    #Shuffle the datasets
    random.shuffle(dataset_files)
    return dataset_files

def load_dataset(dataset_file):
    #Get the current working directory
    cwd = os.getcwd()

    #Define the relative path to the dataset folder and file name using string formatting
    relative_path = 'datasets\\{}'.format(dataset_file)

    #Join the current working directory with the relative path to get the full path to the file
    file_path = os.path.join(cwd, relative_path)

    try:
        #Load the dataset
        data = pd.read_excel(file_path)
        print("Loading", dataset_file)
        return data
    except Exception as e:
        print(f'Error loading {dataset_file}: {str(e)}')
        return None