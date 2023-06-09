import sys
sys.path.append('./')
from mylibs import *

from architectures.LSTM import LSTM
from architectures.CNNAttention import CNNAttention

class Experiment:
    def create_model(self, hparams):
        """
        returns a model based on the selected hyperparameters
        """

        if hparams['HP_ARCHITECTURE']=="CNNAttention":
            return CNNAttention(num_units= hparams['HP_NUM_UNITS'],
                                    num_layers=hparams['HP_NUM_LAYERS'],
                                    num_conv_layers=hparams['HP_NUM_CONV_LAYERS'],
                                    num_heads=hparams['HP_NUM_HEADS'],
                                    attention=['HP_ATTENTION']) 
        
        if hparams['HP_ARCHITECTURE'] == "LSTM":
            return LSTM(num_hidden_units = hparams['HP_NUM_HIDDEN_UNITS'], 
                        num_lstm_layers=hparams['HP_NUM_LSTM_LAYERS'],
                        num_dense_units = hparams['HP_NUM_DENSE_LAYERS'],
                        num_dense_layers = hparams['HP_NUM_DENSE_UNITS'],
                        num_conv_layers=hparams['HP_NUM_CONV_LAYERS'],
                        increase=hparams['HP_INCREASE_UNITS_PER_LSTM_LAYER'],
                        bidirectional=hparams['HP_BIDIRECTIONAL'])


    def run_model(self, train, val, test, hparams, logdir, savedir, checkpointdir, metrics, epochs):
        """
        builds, compiles, trains and evaluates a model with certain architectual hyperparameters
    
        Args:
            hparams: selected hyperparameters
            logdir: directory for logs
            savedir: directors for saving the model
            checkpointdir: direcotry for model checkpoints
            metrics: a list of metrics we want to track
    
        Returns:
            accuracy of trained model evaluated on test data set
        """
        model = self.create_model(hparams)
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics)
    
    
        model.fit(train,
                  batch_size=64,
                  epochs=epochs,
                  validation_data=val,
                  verbose=0, # no output during training
                  callbacks=[tf.keras.callbacks.TensorBoard(logdir),  # log metrics
                            hp.KerasCallback(logdir, hparams),  # log hparams
                            tf.keras.callbacks.ModelCheckpoint(filepath= os.path.join(checkpointdir, "ckpt_{epoch}") ,monitor='val_loss',save_weights_only=True), # save checkpoints when val loss goes down
                            keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)], # early stopping in the case that loss doesnt go down for 3 epochs
                  ) 
        results = model.evaluate(test)
        model.save(savedir)
        return results
    

    def run_experiment(self, hparams_dict, logdir, metrics, train, val, test, epochs=1):
        """
        train models on the selected hyperparamers and log the metrics, checkpoints and trained model in logidr
        """
        print(hparams_dict['HP_ARCHITECTURE'])

        if hparams_dict['HP_ARCHITECTURE'].domain.values[0]=="CNNAttention":
            print('transformer')
            session_num = 0

            # iterate through selected hyperparameters
            for num_units in hparams_dict['HP_NUM_UNITS'].domain.values:
                for num_layers in hparams_dict['HP_NUM_LAYERS'].domain.values:
                    for num_heads in hparams_dict['HP_NUM_HEADS'].domain.values:
                        for num_conv_layers in hparams_dict['HP_NUM_CONV_LAYERS'].domain.values:
                            for attention in hparams_dict['HP_ATTENTION'].domain.values:
                                for architecture in hparams_dict['HP_ARCHITECTURE'].domain.values:
                                    
                                    # build directionary to run model
                                    hparams = {
                                        'HP_NUM_UNITS': num_units,
                                        'HP_NUM_LAYERS': num_layers,
                                        'HP_NUM_HEADS': num_heads,
                                        'HP_NUM_CONV_LAYERS': num_conv_layers,
                                        'HP_ATTENTION': attention,
                                        'HP_ARCHITECTURE': architecture
                                    }
                                    run_name = "run-%d" % session_num
                                    print(type(h) for h in hparams)
                                    print(type(hparams(h) for h in hparams))
                                    print('--- Starting trial: %s' % run_name)
                                    print({h: hparams[h] for h in hparams})

                                    # Run a single experiment
                                    accuracy = self.run_model(train=train,
                                        val=val,
                                        test=test,
                                        hparams= hparams,
                                        epochs = epochs,
                                        logdir=logdir+'/hparam_tuning/' + run_name, 
                                        savedir=logdir+'logs/models/'+run_name, 
                                        checkpointdir=logdir+'logs/checkpoints'+run_name,
                                        metrics=metrics)
                                    session_num += 1


        if hparams_dict['HP_ARCHITECTURE'].domain.values[0]=="LSTM":
            session_num = 0
            # iterate through selected hyperparameters
            for num_lstm_layers in hparams_dict['HP_NUM_LSTM_LAYERS'].domain.values:
                for num_hidden_units in hparams_dict['HP_NUM_HIDDEN_UNITS'].domain.values:
                    for num_conv_layers in hparams_dict['HP_NUM_CONV_LAYERS'].domain.values:
                        for num_dense_layers in hparams_dict['HP_NUM_DENSE_LAYERS'].domain.values:
                            for num_dense_units in hparams_dict['HP_NUM_DENSE_UNITS'].domain.values:
                                for increase in hparams_dict['HP_INCREASE_UNITS_PER_LSTM_LAYER'].domain.values:
                                    for model_architecture in hparams_dict['HP_ARCHITECTURE'].domain.values:
                                        for bidirectional in hparams_dict['HP_BIDIRECTIONAL'].domain.values:
                                            
                                            # build directionary to run model
                                            hparams = {
                                                'HP_ARCHITECTURE': model_architecture,
                                                'HP_NUM_LSTM_LAYERS': num_lstm_layers,
                                                'HP_NUM_HIDDEN_UNITS': num_hidden_units,
                                                'HP_NUM_CONV_LAYERS': num_conv_layers,
                                                'HP_NUM_DENSE_LAYERS': num_dense_layers,
                                                'HP_NUM_DENSE_UNITS': num_dense_units,
                                                'HP_INCREASE_UNITS_PER_LSTM_LAYER': increase,
                                                'HP_BIDIRECTIONAL': bidirectional
                                            }
                                            run_name = "run-%d" % session_num
                                            print('--- Starting trial: %s' % run_name)
                                            print(type(h) for h in hparams)
                                            print(type(hparams[h] for h in hparams))
                                            print({h: hparams[h] for h in hparams})
                                            results = self.run_model(
                                                train=train,
                                                val=val,
                                                test=test,
                                                logdir=logdir+'hparam_tuning/' + run_name, 
                                                hparams=hparams, 
                                                epochs=epochs,
                                                savedir=logdir+'models/'+run_name, 
                                                checkpointdir=logdir+'checkpoints'+run_name,
                                                metrics=metrics)
                                            session_num += 1
                                            