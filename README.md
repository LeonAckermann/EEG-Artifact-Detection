# EEG-Artifact-Detection

This is the repository of the project EEG Artifact Detection for the course Implementing Aritificial Neural Networks with Tensorflow.

## Setup

1. Clone the repository with `git clone https://github.com/LeonAckermann/EEG-Artifact-Detection.git`

2. Create new conda environment `conda create --name myenv`

3. Activate environment `conda activate myenv`

4. Install all necesary packages by navigating to the directory of the cloned repository. Then execute `pip install -r setup/requirements.txt`

5. Download the necessary data with `bash setup/data_download.sh`

## Structure of repository

### Data
Firstly, in the [data folder](data) you will find a [python module](data/data.py) with all necessary functions for the preprocessing of the data. In the [notebook](data/data_statistics_visualisations.ipynb) you can inspect data samples and statistics about the data.

### Architecture
In the folder [Architectures](architectures) you will find the implementations of our models. We implemented a [Convlutional Transformer based network](architectures/CCNAttentionNetwork.py), a [Multi Layer LSTM](architectures/MultiLayerLSTM.py) and [Bidirectional LSTM](architectures/BidirectionalLSTM.py).

### Hyperparameter Optimization
In the folder [hparams_optimization](hyerparams_optimization) you find two jupyternotebooks for the optimization of the [transformer architecture](hyperparams_optimization/transformer_architecture_tuning.ipynb) and [lstm architectures](hyperparams_optimization/lstm_architecture_tuning.ipynb).


### bash scripts

Lastly, in the folder [bash scripts](bash_scripts), you can find useful bash scripts to [setup a ssh connection](bash_scripts/vast_setup.md) to a cloud gpu, [copy logs](bash_scripts/copy_remote_to_local.sh) from the cloud server to your local machine and [upload](bash_scripts/save_logs_to_tensorboard.sh) to tensorboard
