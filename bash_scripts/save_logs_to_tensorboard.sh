pip install tensorboard

tensorboard dev upload \
    --logdir ./logs/hparam_tuning_architecture \
    --name "Artifact Detection 7" \
    --description "first try hyperparameter optimization of architecture" \
    --one_shot