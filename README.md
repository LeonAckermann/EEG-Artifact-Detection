# EEG-Artifact-Detection

This is the repository of the project EEG Artifact Detection for the course Implementing Aritificial Neural Networks with Tensorflow.

## Setup

1. Clone the repository with `git clone https://github.com/LeonAckermann/EEG-Artifact-Detection.git`

2. Create new conda environment `conda create --name myenv`

3. Activate environment `conda activate myenv`

4. Install all necesary packages by navigating to the directory of the cloned repository. Then execute `pip install -r requirements.txt`

## Structure of repository

### Architecture
In the folder [Architectures](architectures) you will find the implementations of our models. We implemented a [Convlutional Transformer based network](architectures/CCNAttentionNetwork.py), a [Multi Layer LSTM](architectures/MultiLayerLSTM.py) and [Bidirectional LSTM](architectures/BidirectionalLSTM.py).
- data_download.sh --> to download the needed data from our google drive
- data.py --> all necessary functions regarding the data
- data_statistics.ipynb --> a notebook with plots and helpful data statistics

In the folder "architectures" you can find all model architectures that we tried out and optimized which are the following:
- Attention Model
- MultiLayer LSTM
- Bidirectional LSTM
- Residual LSTM
- Convolutional Model

In the "hyperparams_optimization" folder you can find the different notebooks we made use of to optimize the hyperparameters of our models. The folder is subcategorised into hyperparameter otpimization of architecture and learning/regularization. This is the structure:
- Architectual Optimization
    - recurrence based models
    - attention based models
- Learning optimization
    - recurrence based models
    - attention based models

Lastly, in the folder "bash script", we provide how we worked with remote GPU servers to log and save our results
