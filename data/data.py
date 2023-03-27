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
    
    def prepare_data(input, balance=False, dataset=False, batch_size=64):

        print(input.shape)
        # rearrange data so that the channels are in the last dimension (following convention)
        #data = einops.rearrange(input, 'b c t -> b t c')
        #features = data[:,:,:-2] # only take inputs from first 17 channels and all timesteps
        #labels = data[:,:,-2:].astype('int32') # targets are stored in last two channels
#
        #if balance:
        #    idx_ones = set(np.where(labels.any(axis=1))[0])
        #    features = features[np.array(list(idx_ones))]
        #    labels = labels[np.array(list(idx_ones))]
    #
        #if dataset:
        #    dataset = tf.data.Dataset.from_tensor_slices((features, labels)) # create dataset
        #    dataset = dataset.map(lambda x,y: (x, tf.cast(y, tf.int32)))
        #    dataset = dataset.cache().shuffle(1000).batch(batch_size).prefetch(20) # as usual
        #    return dataset
#
        #return features, labels
    
    
    def plot(self, data):
        # code to plot the data
        pass
