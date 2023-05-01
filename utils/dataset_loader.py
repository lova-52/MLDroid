import random

def load_dataset():
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
