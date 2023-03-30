import sys
sys.path.append('./')
from mylibs import *

class Data:
    def load(self, data_path):
        """
        load pickle file from filepath and return it
        """
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def split(self, data, train=0.8, val=0.1, test=0.1):
        """
        split the data into three sets namely training, validation and test

        Args:
            data: np.array
            train: float
            val: float
            test: float

        Returns: 
            A List of with the three datasets
        """

        train_split_idx = int(data.shape[0]*0.8) # get index for split between train and val
        val_split_idx = train_split_idx + int(data.shape[0]*0.1) # get index for split between val and test
        train_ds = data[:train_split_idx, :, :] # split train
        val_ds = data[train_split_idx:val_split_idx, :, :] # split val
        test_ds = data[val_split_idx:, :, :] # split test
        return train_ds, val_ds, test_ds
    
    def prepare_data(self, input, balance=False, dataset=False, batch_size=64, buffer_size= 20, transformer=False, lstm=False, ccn=False, musc=False, eyem=False):
        """
        split the data into features and lables, reshape the data, balance the data and put it into tf.Dataset

        Args:
            input: np.array
            balance: bool --> if ture function returns balanced data
            dataset: bool --> if true function returns dataset, if false np.array is returned
            batch_size: int --> determines batch_size if dataset is returned
            buffer_size: int --> size for prefetching data set
            transformer: bool --> to create dataset for transformer
            lstm: bool --> to create dataset for lstm
        
        Returns: 
            if dataset==true --> tensorflow dataset is returned with shape (features, labels) for indicated architecture type
            if dataset==false --> features and labels are returned seperately as np.array for indicated architecture type
        """        

        if transformer+lstm== 0:
            print('Please indicate for which architecture type you want to prepare the data with the arguements transformer=1 or lstm=1')
            return
        
        # rearrange data so that the channels are in the last dimension (following convention)
        data = einops.rearrange(input, 'b c t -> b t c')
        features = data[:,:,:-2] # only take inputs from first 17 channels and all timesteps


        if lstm:
            labels_both = data[:,:,-2:]
            labels_muscle = data[:,:,-2].astype('int32') # targets are stored in last two channels
            labels_eyem = data[:,:,-1].astype('int32')

            if balance:
                indices_1 = np.where(np.any(labels_muscle == 1, axis=1))[0]
                indices_2  = np.where(np.any(labels_eyem == 1, axis=1))[0]
                idx_ones = np.union1d(indices_1, indices_2)
                #idx_ones = set(np.where(labels.any(axis=1))[0])
                #features = features[np.array(list(idx_ones))]
                #labels = labels[np.array(list(idx_ones))]
                
                features = features[idx_ones]
                labels_muscle = labels_muscle[idx_ones]
                labels_eyem = labels_eyem[idx_ones]
                labels_both = labels_both[idx_ones]

            if musc:
                labels = labels_muscle
            elif eyem:
                labels = labels_eyem
            else:
                labels = labels_both

            if ccn:
                features = tf.expand_dims(features, axis=-1)

            if dataset:
                dataset = tf.data.Dataset.from_tensor_slices((features, labels)) # create dataset
                dataset = dataset.map(lambda x,y: (x, tf.cast(y, tf.int32)))
                dataset = dataset.cache().shuffle(1000).batch(batch_size).prefetch(buffer_size) # as usual
                return dataset
            
            return features, labels


        if transformer:
            y_data_1 = data[:,:,-1] 
            y_data_2 = data[:,:,-2]  

            if balance:
                # if we want to reduce the dataset to only the samples that contain positive examples:
                indices_1 = np.where(np.any(y_data_1 == 1, axis=1))[0]
                indices_2  = np.where(np.any(y_data_2 == 1, axis=1))[0]
                indices_comb = np.union1d(indices_1, indices_2)

                # Extract the data and labels corresponding to these indices
                features = features[indices_comb]
                y_data_1 = y_data_1[indices_comb]
                y_data_2 = y_data_2[indices_comb] 

            if dataset:
                # create a tensorflow dataset 
                ds = tf.data.Dataset.from_tensor_slices((features, y_data_1, y_data_2))

                ds = ds.map(lambda x,y1, y2: (x, tf.cast(y1, tf.int32), tf.cast(y2, tf.int32)))
                ds = ds.shuffle(1000).batch(batch_size).cache().prefetch(buffer_size = buffer_size)
                return ds
            
            return features, y_data_1, y_data_2
    
    
    def plot(self, data, save_name=None, show=True, annotate=False, artifact=None):
        data = np.squeeze(data)

        if annotate:
            names = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'FZ', 'CZ', 'PZ']

            if artifact is not None:
                # if artifact is a list 
                if isinstance(artifact, list):
                    for i in artifact:
                        names.append(i)
                else:
                    names = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'FZ', 'CZ', 'PZ', artifact]

            if data.shape[0] == len(names):
                fig = plt.figure(figsize=(20,10), dpi=200)
                for i in range(len(names)):
                    plt.subplot(len(names), 1, i+1)
                    plt.plot(data[i, :])
                    if annotate:
                        plt.annotate(names[i], xy=(1.05, 0.5), xycoords='axes fraction', horizontalalignment='right', verticalalignment='center',fontsize=4)
            elif data.shape[-1] == len(names):
                fig = plt.figure(figsize=(20,10), dpi=200)
                for i in range(len(names)):
                    plt.subplot(len(names), 1, i+1)
                    plt.plot(data[:, i])
                    if annotate:
                        plt.annotate(names[i], xy=(1.05, 0.5), xycoords='axes fraction', horizontalalignment='right', verticalalignment='center', fontsize=4)

        else:
            fig = plt.figure(figsize=(20,10), dpi=200)
            for i in range(len(data)):
                plt.subplot(len(data), 1, i+1)
                plt.plot(data[i, :])
                if annotate:
                    plt.annotate(names[i], xy=(1.05, 0.5), xycoords='axes fraction', horizontalalignment='right', verticalalignment='center', fontsize=4)


        if save_name:
            plt.savefig(save_name)
        if show:
            plt.show()


    def get_balance(self, labels):
        """
        calculate the balance between timesteps with artifacts and without
        """
        ones = np.argwhere(labels==1).size
        zeros = np.argwhere(labels==0).size
        total = ones + zeros
        print('Balance:\n    Count of Timesteps with no artifacts at all: {}\n    Count Timesteps with artifacts: {} ({:.2f}% of total)\n'.format(
            zeros, ones, 100 * (ones) / total))
        
    def get_artifact_duration(self, data):
        """
        calculates the duration of muscel artifact and eye movement artifact in hours
        """

        muscel_timesteps = np.argwhere(data[:,-2,:]==1).size 
        eyem_timesteps = np.argwhere(data[:,-1, :]==1).size 
        muscel_hours = muscel_timesteps / (128*60*60) 
        eyem_hours = eyem_timesteps / (128*60*60)
        print('Number of events muscel artifacts: {}\nNumber of events eye movement artifacts: {}'.format(muscel_timesteps,eyem_timesteps))
        print('Hours of muscel artifacts: {}\nHours of eye movement artifacts: {}'.format(muscel_hours, eyem_hours))