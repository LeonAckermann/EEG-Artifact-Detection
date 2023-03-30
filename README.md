# EEG-Artifact-Detection

This is the repository of the project EEG Artifact Detection for the course Implementing Aritificial Neural Networks with Tensorflow. Below you can find an overview of the contents of this repository and the precise steps for replicating our results either on your local machine or a vast.ai gpu.

## Structure of Repository

### Data
Firstly, in the [data folder](data) you will find a [python module](data/data.py) with all necessary functions for downloading the data from our drive and to perform the necessary preprocessing steps. In the [notebook](data/data_statistics_visualisations.ipynb) you can inspect data samples and statistics about the data.

### Architecture
In the folder [Architectures](architectures) you will find the implementations of our models. We implemented a [Convlutional Transformer based network](architectures/CCNAttentionNetwork.py), a [Multi Layer LSTM](architectures/MultiLayerLSTM.py) and [Bidirectional LSTM](architectures/BidirectionalLSTM.py).

### Hyperparameter Optimization
In the folder [hparams_optimization](hyerparams_optimization) you find two jupyternotebooks for the optimization of the [transformer architecture](hyperparams_optimization/transformer_architecture_tuning.ipynb) and [lstm architectures](hyperparams_optimization/lstm_architecture_tuning.ipynb).


### Bash Scripts

Lastly, in the folder [bash scripts](bash_scripts), you can find useful bash scripts to [setup a ssh connection](bash_scripts/vast_setup.md) to a cloud gpu, [copy logs](bash_scripts/copy_remote_to_local.sh) from the cloud server to your local machine and [upload](bash_scripts/save_logs_to_tensorboard.sh) your training logs to tensorboard.

## Setup for Local Machine

1. Clone the repository with `git clone https://github.com/LeonAckermann/EEG-Artifact-Detection.git`

2. Create new conda environment `conda create --name myenv`

3. Activate environment `conda activate myenv`

4. Install all necesary packages by navigating to the directory of the cloned repository. Then execute `pip install -r setup/requirements.txt`

5. Download the necessary data with `bash setup/data_download.sh`

6. Run either of the hyperparam optimization scripts for [transformer](hyperparams_optimization/transformer_architecture_tuning.ipynb) or [lstms](hyperparams_optimization/lstm_architecture_tuning.ipynb)

7. Keep in mind that without an external GPU, your kernel might crash

## Setup on Vast.ai 

In case you'd like to train our models for the purpose of reproducing our findings, you'll most likely need a GPU. The following instructions will guide you through
the setup of a virtual machine over the popular cloud computing provider vast.ai. 

1. Generate ssh keypair for connection with server by executing `ssh-keygen -t rsa -b 4096``

2. Display the generated public key `cat ~/.ssh/id_rsa.pub`

3. Copy the contents

4. Past Content in Vast.ai public key field

5. Rent GPU and connect over jupyter notebook

6. Clone the repository with `git clone https://github.com/LeonAckermann/EEG-Artifact-Detection.git`

7. Install all packages by running `bash setup/requirements.sh`

8. Download the necessary data with `bash setup/data_download.sh`

9. Run either of the optimization notebooks

10. when finished with the model training, we can copy our logs and models to our local machine with the following command `scp -r -P <port> root@<ip-adress>:./logs ./Desktop`



