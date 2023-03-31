import numpy as np
import pyedflib
import os
import tqdm
from scipy.stats import mode
import itertools
import pandas as pd
import sys
sys.path.append('../EEG2023/')

from utils.preprocessing import normalize, z_score, convert_to_chunks, remove_samples

def load(mode = 'channel_prediction', summary = False, normalize_data = True, z_score_data = True, seconds_per_sample = 300, target_frequency = 1, noise_level = 0, padding = 0, folder = None, artifact = None, exclude_last_channels = None, seed = 23):
    """ Function to load and preprocess the data from the EDF files.
    
    Parameters
    ----------
    mode : str, optional
        The mode to load the data in. The default is 'channel_prediction'.
    summary : bool, optional
        If True, a summary of the data will be printed. The default is False.
    normalize_data : bool, optional
        If True, the data will be normalized. The default is True.
    z_score_data : bool, optional
        If True, the data will be z-scored. The default is True.
    seconds_per_sample : int, optional
        The number of seconds per sample. The default is 300.
    target_frequency : int, optional
        The target frequency to downsample the data to. The default is 1.
    noise_level : int, optional
        The noise level to add to the data. The default is 0.
    padding : int, optional
        The number of seconds to pad the data with. The default is 0.
    folder : str, optional
        The folder to load the data from. The default is None.
    artifact : str, optional
        The type of artifact to add to the data. The default is None.
    exclude_last_channels : int, optional
        The number of channels to exclude from the end of the data. The default is None.
    nine_channel_config : bool, optional
        If True, the data will be loaded in the 9 channel configuration. The default is False.
    seed : int, optional
        The seed to use for the random number generator. The default is 23.
            
    Returns 
    ----------
    raw : The preprocessed data with shape (n_samples, n_channels, n_timepoints).
        
    """

    if mode == 'channel_prediction':

        if folder is not None:
            path = folder
        else:
            path = 'data/edf/'

        # Print summary of data
        if summary:
            print('')
            print('Summary:')
            for key, value in data_summary(path)[0].items():
                print(key, value)
            print('')

        common_channels = data_summary(path)[1]

        # Change common channels list to only have entries that have EEG in the name
        common_channels = [x for x in common_channels if x.__contains__('EEG')]

        # Discard the rest of the word after the first "-"
        common_channels = [x.split('-')[0] for x in common_channels]

        edfs = []
        for i in tqdm.tqdm([x for x in os.listdir(path) if x.endswith('.edf')]):

            # Create Y
            if artifact is not None:
                # if artifact argument is a list
                if isinstance(artifact, list):
                    y_arr = []
                    for k in artifact:
                        y_arr.append(mark_artifact(path + i, k))
                    y_arr = np.vstack(y_arr)

                # if artifact is a string
                else:
                    y_arr = mark_artifact(path + i, artifact)

            f = pyedflib.EdfReader(path + i)
            raw_ = np.empty((0, f.getNSamples()[0]))

            # Check if the number of channels in the file is equal to the number of common channels
            if f.signals_in_file < len(common_channels):
                continue

            j = [x for x in f.getSignalLabels() if x.split('-')[0] in common_channels][0:len(common_channels)]

            if len(j) != len(common_channels):
                print('Error: ', i, '\n', len(j), len(common_channels))
                continue
            
            # Find the places in the original .getSignalLabels() function where the channels of interest are
            k = [f.getSignalLabels().index(x) for x in j]

            # Read the signals
            for i in k:
                signal = f.readSignal(i)
                raw_ = np.vstack((raw_, signal))

            # Get the sample frequency
            sf = f.getSampleFrequency(0)

            # Normalize and z-score the data
            if normalize_data:
                raw = normalize(raw_)
                assert raw.shape == raw_.shape
            if z_score_data:
                raw = z_score(raw, 100, visualize=False)
                assert raw.shape == raw_.shape

            raw = np.vstack((raw, y_arr))

            # Convert to chunks
            raw = convert_to_chunks(raw, int(sf), seconds_per_sample, target_freq=target_frequency, noise_level=noise_level, padding=padding, seed=seed, exclude_last_channels=exclude_last_channels)

            # Append to edfs
            edfs.append(raw)

            f.close()

        raw_ = np.vstack(edfs)
        raw = remove_samples(raw_)

        if artifact is not None:
            # if artifact is a list
            if isinstance(artifact, list):
                ones_count_1 = []
                for i in range(raw.shape[0]):
                    for j in range(raw.shape[2]):
                        if raw[i, -2, j] != 0:
                            ones_count_1.append(i)
                            continue

                ones_count_2 = []
                for i in range(raw.shape[0]):
                    for j in range(raw.shape[2]):
                        if raw[i, -1, j] != 0:
                            ones_count_2.append(i)
                            continue

                print('\nLoading:\n', raw.shape, '\n', 'Bad samples:', raw_.shape[0] - raw.shape[0], 
                    '\n', 'Total hours:', round(raw.shape[0] * raw.shape[2] / 60 / 60 / target_frequency, 3), 
                    '\n', f'Total {artifact[0]} samples:', len(np.unique(ones_count_1)),
                    '\n', f'Total {artifact[1]} samples:', len(np.unique(ones_count_2)), '\n')

            # if artifact is a string
            else:
                ones_count = []
                for i in range(raw.shape[0]):
                    for j in range(raw.shape[2]):
                        if raw[i, -1, j] != 0:
                            ones_count.append(i)
                            continue

                print('\nLoading:\n', raw.shape, '\n', 'Bad samples:', raw_.shape[0] - raw.shape[0], 
                    '\n', 'Total hours:', round(raw.shape[0] * raw.shape[2] / 60 / 60 / target_frequency, 3), 
                    '\n', f'Total {artifact} samples:', len(np.unique(ones_count)), '\n')
        else:
            print('\nLoading:\n', raw.shape, '\n', 'Bad samples:', raw_.shape[0] - raw.shape[0], 
                    '\n', 'Total hours:', round(raw.shape[0] * raw.shape[2] / 60 / 60 / target_frequency, 3), '\n')

        return raw

def data_summary(directory):
    """ Function to summarize the data in a directory.
    
    Parameters
    ----------
    directory : str
        The directory containing the data.

    Returns 
    ----------
    mean_sf_values : list
        The mean sampling frequency of the data.
    mode_sf_values : list
        The mode sampling frequency of the data.
    duration_values : list
        The duration of the data in seconds.
    common_channels : list
        The common channels of the data.
    channel_set : set
        The set of all channels in the data.
    total_duration : list
        The total duration of the data in hours.
        
    """

    mean_sf_values = []
    mode_sf_values = []
    duration_values = []
    common_channels = []
    channel_set = set()
    total_duration = []

    for i in [x for x in os.listdir(directory) if x.endswith('.edf')]:
        file_path = os.path.join(directory, i)
        edf_file = pyedflib.EdfReader(file_path)

        # Mean sampling frequency
        signal_freq = edf_file.getSampleFrequency(0)
        mean_sf_values.append(signal_freq)

        # Mode sampling frequency (same as mean since there is only one frequency)
        mode_sf_values.append(signal_freq)

        # Duration in seconds
        duration = edf_file.getFileDuration()
        duration_values.append(duration)

        # Common channels
        channels = [x.split('-')[0] for x in edf_file.getSignalLabels()]
        common_channels.append(set(channels))

        # Set of channels that are in all files
        if not channel_set:
            channel_set.update(channels)
        else:
            channel_set.intersection_update(channels)

        # Get the measured seconds
        measured_seconds = edf_file.getFileDuration()
        total_duration.append(measured_seconds)

        edf_file.close()

    summary_stats = {
        'mean_sf': np.mean(mean_sf_values),
        'mode_sf': mode(np.asarray(mode_sf_values, dtype=object), keepdims=False),
        'min_duration': np.min(duration_values),
        'max_duration': np.max(duration_values),
        'mean_duration': np.mean(duration_values),
        'std_duration': np.std(duration_values),
        'common_channels': sorted(list(channel_set)),
        'num_common_channels': len(channel_set),
        'total_duration': np.sum(total_duration) / 3600,
    }

    return summary_stats, sorted(list(channel_set))

def mark_artifact(filename, artifact):
    """ Function to mark the artifact from the annotated CSV file.
    
    Parameters
    ----------
    filename : str
        The name of the EDF file.
    artifact : str
        The name of the artifact to be marked.
            
    Returns 
    ----------
    artifact_array : numpy array
        A numpy array of size (1, length) with 1's marking the artifact.
        
    """

    # read the EDF file and get its length
    f = pyedflib.EdfReader(filename)
    length = f.getNSamples()[0]
    f.close()
    
    # read the CSV file with shiver annotations
    csv_filename = filename[:-4] + ".csv"
    df = pd.read_csv(csv_filename, skiprows=6)
    
    # create an array of zeros of the same length as the EDF file
    artifact_array = np.zeros(length) 
    
    # mark shivers as 1's in the array
    for index, row in df[df['label']==artifact].iterrows():
        start = int(row['start_time'] * f.getSampleFrequency(0))
        stop = int(row['stop_time'] * f.getSampleFrequency(0))
        artifact_array[start:stop] = 1
    
    # reshape the array to be of size (1, length)
    artifact_array = artifact_array.reshape((1, length))
    
    return artifact_array