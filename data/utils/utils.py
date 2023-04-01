import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import pickle
import zipfile
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def to_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'Pickled: {os.path.getsize(filename) / 1e6} MB / {os.path.getsize(filename) / 1e9} GB')

def to_zip(pickle_filename, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pickle_filename)
    print(f'Zipped: {os.path.getsize(zip_filename) / 1e6} MB / {os.path.getsize(zip_filename) / 1e9} GB')

def from_zip(zip_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        zipf.extractall()
    print(f'Unzipped: {os.path.getsize(zip_filename) / 1e6} MB / {os.path.getsize(zip_filename) / 1e9} GB')
        
def from_pickle(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        arr = pickle.load(f)
        print(f'Loaded: {arr.shape}')
        return arr

def pickle_check(pickle_filename):
    print('')
    print('Checking pickle and zip functions...')
    os.system(f'cp {pickle_filename} test.pkl')
    data = from_pickle('test.pkl')
    to_zip('test.pkl', 'test.zip')
    from_zip('test.zip')
    loaded_data = from_pickle('test.pkl')
    assert np.array_equal(data, loaded_data)
    os.remove('test.zip')
    os.remove('test.pkl')
    print('Success!\n')

def split_files(folder_path, num_subfolders):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
    num_files = len(file_list)

    # Calculate the number of files to put in each subfolder
    files_per_subfolder = int(num_files / num_subfolders)
    
    # Create the subfolders if they don't already exist
    subfolder_names = [f"{folder_path}/{i}of{num_subfolders}" for i in range(1, num_subfolders + 1)]
    for subfolder_name in subfolder_names:
        os.makedirs(subfolder_name, exist_ok=True)
    
    # Copy the files into the subfolders
    progress_bar = tqdm(total=num_files)
    subfolder_idx = 0
    num_copied = 0
    for file_name in file_list:
        source_path = os.path.join(folder_path, file_name)
        destination_folder = subfolder_names[subfolder_idx]
        destination_path = os.path.join(destination_folder, file_name)
        shutil.move(source_path, destination_path)
        num_copied += 1
        progress_bar.update(1)
        if num_copied == files_per_subfolder:
            num_files_in_subfolder = len(os.listdir(destination_folder))
            print(f"{destination_folder}: {num_files_in_subfolder}")
            if subfolder_idx < num_subfolders - 1:
                subfolder_idx += 1
                num_copied = 0
    
    progress_bar.close()

def load_pickled_data(root_dir = 'data/training', file_name=None, folder_name='300'):
    data = []
    if file_name is None:
        path = os.path.join(root_dir, f"{folder_name}")

        if len(os.listdir(path)[0].split('_')) > 3:
            frequency = int(os.listdir(path)[0].split('_')[2][:-2])
        else:
            frequency = 1

        with open(os.path.join(path, file_name), "rb") as f:
            data.append(pickle.load(f))
    else:
        if file_name.endswith(".pkl"):
            with open(file_name, "rb") as f:
                data.append(pickle.load(f))
        
        if len(file_name.split('_')) > 3:
            frequency = int(file_name.split('_')[2][:-2])
        else:
            frequency = 1

    data = np.vstack(data)
    print('\nLoading:\n', data.shape, '\n', 'Total hours:', round(data.shape[0] * data.shape[2] / 60 / 60 / frequency, 3), '\n')
    return data

def save_to_pickle_multiple(data, num_files=10, target_path="."):
    os.makedirs(target_path, exist_ok=True)

    samples_per_file = data.shape[0] // num_files
    remainder = data.shape[0] % num_files
    for i in tqdm(range(num_files)):
        start_idx = i * samples_per_file
        end_idx = (i + 1) * samples_per_file
        if i == num_files - 1:
            end_idx = data.shape[0]
        if i < remainder:
            end_idx += 1
        subset_data = data[start_idx:end_idx]
        with open(os.path.join(target_path, f'data_{data.shape[2]}_1hz_{i+1}.pkl'), 'wb') as f:
            pickle.dump(subset_data, f)

def plot(data, save_name=None, show=False, annotate=False):
    data = tf.squeeze(data)

    names = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'FZ', 'CZ', 'PZ']

    if data.shape[0] == 9:
        fig = plt.figure(figsize=(10,5), dpi=100)
        for i in range(9):
            plt.subplot(9, 1, i+1)
            plt.plot(data[i, :])
            if annotate:
                plt.annotate(names[i], xy=(1.05, 0.5), xycoords='axes fraction', horizontalalignment='right', verticalalignment='center')
    elif data.shape[-1] == 9:
        fig = plt.figure(figsize=(10,5), dpi=100)
        for i in range(9):
            plt.subplot(9, 1, i+1)
            plt.plot(data[:, i])
            if annotate:
                plt.annotate(names[i], xy=(1.05, 0.5), xycoords='axes fraction', horizontalalignment='right', verticalalignment='center')
    if save_name:
        plt.savefig(save_name)
    if show:
        plt.show()