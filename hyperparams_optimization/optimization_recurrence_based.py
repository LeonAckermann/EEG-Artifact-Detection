import sys
sys.path.append('./')
from mylibs import *

HP_MODEL_ARCHITECTURE = hp.HParam('model_architecture', hp.Discrete(['lstm1', 'lstm2']))
HP_NUM_HIDDEN_UNITS = hp.HParam('num_hidden_units', hp.Discrete([16, 32, 64, 128, 256, 512]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
#HP_LEARNING_RATE = hp.HParam('learning rate', hp.Discrete([1e-2, 1e-3, 1e-4]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_MODEL_ARCHITECTURE,HP_NUM_HIDDEN_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

metrics = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.CategoricalAccuracy(name='accuracy'),
      keras.metrics.CategoricalCrossentropy(name='loss'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

# method to iterate through various architectures
def build_model(hparams):
    models = {
        "lstm1": LSTM1(hidden_units = hparams[HP_NUM_HIDDEN_UNITS], 
                       activation_function='relu'),
        
        "lstm2": LSTM2(hidden_units=hparams[HP_NUM_HIDDEN_UNITS], 
                       activation_function='relu', 
                       dropout=hparams[HP_DROPOUT])
    }

    return models.get(hparams[HP_MODEL_ARCHITECTURE])


# build, compile and fit the model based on hyperparameters
def run(hparams, metrics,logdir):
  model = build_model(hparams)
  model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=metrics,
  )


  model.fit(train_features[:2,:,:], 
            train_labels[:2,:,:],
            #batch_size=2,
            epochs=1,
            callbacks=[tf.keras.callbacks.TensorBoard(logdir),  # log metrics
                      hp.KerasCallback(logdir, hparams),  # log hparams
                       ],
            ) 
  results = model.evaluate(test_features[:2,:,:], test_labels[:2,:,:])
  return results


session_num = 0
for model_architecture in HP_MODEL.domain.values:
  for num_units in HP_NUM_HIDDEN_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
      for optimizer in HP_OPTIMIZER.domain.values:
        hparams = {
            HP_MODEL_ARCHITECTURE: model_architecture,
            HP_NUM_HIDDEN_UNITS: num_units,
            HP_DROPOUT: dropout_rate,
            HP_OPTIMIZER: optimizer,
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        results = run(logdir='logs/hparam_tuning/' + run_name, hparams=hparams, metrics=metrics)
        session_num += 1