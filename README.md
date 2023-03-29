# EEG-Artifact-Detection

This is the repository of the project EEG Artifact Detection for the course Implementing Aritificial Neural Networks with Tensorflow.

## Setup

Clone the repository with 
```bash
git clone https://github.com/LeonAckermann/EEG-Artifact-Detection.git
```

**Create new conda environment** <br>
```bash
conda create --name myenv
```

**Activate environment**
```bash
conda activate myenv
```

**Install necessary packages** <br>
Navigate to the clones repository on your local machine. Execute the following command to install all necessary packages.
```bash
pip install -r requirements.txt
```


## Structure of repository

In the folder "data" you can find
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
