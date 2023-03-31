from random import seed
import tensorflow as tf
import numpy as np
import os
import einops
from scipy.signal import resample
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def normalize(data):
    data = (data - data.mean()) / data.std()
    return data

def z_score(data, window_size, visualize=False):
    mean_vals = np.mean(data, axis=1, keepdims=True)
    std_vals = np.std(data, axis=1, keepdims=True)
    lower_bound = mean_vals - 3 * std_vals
    upper_bound = mean_vals + 3 * std_vals
    
    mask = np.logical_or(data > upper_bound, data < lower_bound)
    for i in range(len(data)):
        for j in range(window_size, len(data[0]) - window_size):
            if mask[i, j]:
                if visualize:
                    plt.figure(figsize=(20,10))
                    plt.title(f'Channel {i}, Sample {j}')
                    plt.subplot(2, 1, 1)
                    plt.plot(data[i, j-window_size:j+window_size])
                    plt.plot(np.arange(window_size-5, window_size+5), data[i, j-5:j+5], color='red')
                data[i, j] = np.mean(data[i, j-20:j+20])
                if visualize:
                    plt.subplot(2, 1, 2)
                    plt.plot(data[i, j-window_size:j+window_size])
                    plt.legend()
                    plt.show()
    return data

def convert_to_chunks(data, sampling_rate, window_seconds, label=None, debug=False, exclude_last_channels = None, target_freq=1, noise_level=0, padding=0, seed=23):

    if exclude_last_channels is not None:
        last_channel = data[-exclude_last_channels:, :]
        data = data[:-exclude_last_channels, :]
        
    if debug:
        print("Data shape:", data.shape)
        print("sampling rate:", sampling_rate)
        print("target freq:", target_freq)
        print(target_freq / sampling_rate)
        print(int(len(data[0, :]) * (target_freq / sampling_rate)))
    
    new_array = np.zeros((data.shape[0], int(len(data[0, :]) * (target_freq / sampling_rate))))
    for i in range(data.shape[0]):
        new_array[i, :] = resample(data[i, :], int(len(data[i, :]) * (target_freq / sampling_rate)))

    if exclude_last_channels is not None:

        temp = np.zeros((last_channel.shape[0], new_array.shape[1]))
        for i in range(last_channel.shape[0]):
            temp_ = last_channel[i, :]
            f = interp1d(np.arange(len(temp_)), temp_)
            temp_ = f(np.linspace(0, len(temp_)-1, new_array.shape[1]))

            # Threshold the last channel
            temp_[temp_ > 0.5] = 1
            temp_[temp_ <= 0.5] = 0

            temp[i, :] = temp_

        last_channel = temp

    if debug:
        print("Data shape after resampling:", new_array.shape)

    if noise_level > 0:
        np.random.seed(seed)
        for i in range(new_array.shape[0]):
            noise = noise_level * np.random.randn(new_array.shape[1])
            mask = np.random.randint(0, 2, new_array.shape[1]) # So the noise is only added to half of the data
            new_array[i, :] += noise * mask

    data_seconds_averaged = new_array

    window_size = int(window_seconds * target_freq)

    dims = data.shape[0]

    if exclude_last_channels is not None:
        data_seconds_averaged = np.vstack((data_seconds_averaged, last_channel))
        dims += exclude_last_channels
    
    # Split data into windows
    num_windows = data_seconds_averaged.shape[1] // window_size
    data_seconds_split = np.zeros((num_windows, dims , window_size))
    for i in range(num_windows):
        data_seconds_split[i, :, :] = data_seconds_averaged[:, i*window_size:(i+1)*window_size]

    if debug:
        print("Data shape after splitting:", data_seconds_split.shape)

    # Add random padding cutoffs with 0s only at the end of the data if needed
    if padding > 0:
        np.random.seed(seed)
        for i in range(data_seconds_split.shape[0]):
            if np.random.rand() < padding:
                cutoff = np.random.randint(0, int((window_size * 0.8)))
                data_seconds_split[i, :, -cutoff:] = 0
                if debug:
                    print(f"Padding cutoff: #{i} with {cutoff}")

    if label is not None:
        if debug:
            print("Label:", label)
            print("Label chunks shape:", [label] * num_windows)
            print("Data chunks shape:", data_seconds_split.shape)
        return data_seconds_split, [label] * num_windows

    return data_seconds_split

def remove_samples(data, labels = None, debug=False):
    max_vals = np.max(data, axis=-1)
    min_vals = np.min(data, axis=-1)
    mask = np.logical_or(max_vals > 4, min_vals < -4)
    if debug:
        print("Data points larger than 4:", np.where(max_vals > 4))
        print("Data points smaller than -4:", np.where(min_vals < -4))
    mask = np.any(mask, axis=-1)
    if debug:
        print(mask.shape)
    removed_data = data[~mask]

    if labels is not None:
        removed_labels = np.array(labels)[~mask]
        return removed_data, removed_labels

    return removed_data

if __name__ == "__main__":
    from utils import plot

    data = np.random.randn(9, 200000)
    data = convert_to_chunks(data, 100, 300, debug=True, target_freq=128, padding=0.5)
    plot(data[5],show=True)