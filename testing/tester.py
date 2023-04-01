import sys
sys.path.append('./')
from mylibs import *

class Tester:
    
    def load_model(self, logdir, ckptdir=None):
        """
        load a model from the logdir with the best checkpoints if provided
        """
        model = keras.models.load_model(logdir)
        if ckptdir!=None:
            model.load_weights(ckptdir)
        return model
    
    def get_metric(self, metric, label, pred):
        """
        is used for the function get_metrics and computes the passed metric between the label and pred
        """
        m = metric
        m.update_state(label, pred)
        result = m.result().numpy()
        m.reset_state()
        return result
    
    def get_f1(self, precision, recall):
        """
        computes the f1 score based on the passed precision and recall
        """
        return 2*precision*recall*(precision+recall)
    
    def get_metrics(self, model, test_features, test_labels):
        """
        computes accuracy, precision, recall and f1-score for the given model on given test_features, test_labels and predicitions from model
        returns a print statement to read out metrics comfortably
        """
        predictions = model.predict(test_features)
        
        prediction_musc = np.squeeze(predictions[:,:,0])
        prediction_eyem = np.squeeze(predictions[:,:,1])
        test_labels_musc = np.squeeze(test_labels[:,:,0])
        test_labels_eyem = np.squeeze(test_labels[:,:,1])

        # Convert the predicted probabilities to binary labels
        predictions_bin = tf.cast(tf.greater(predictions, 0.5), tf.int32)
        prediction_musc_bin = tf.cast(tf.greater(prediction_musc, 0.5), tf.int32)
        prediction_eyem_bin = tf.cast(tf.greater(prediction_eyem, 0.5), tf.int32)

        # compute accuracies
        accuracy = self.get_metric(tf.keras.metrics.BinaryAccuracy(), test_labels, predictions_bin) # for both artifacts
        accuracy_musc = self.get_metric(tf.keras.metrics.BinaryAccuracy(), test_labels_musc, prediction_musc_bin) # only for muscler artifacts 
        accuracy_eyem = self.get_metric(tf.keras.metrics.BinaryAccuracy(), test_labels_eyem, prediction_eyem_bin) # only for eye movement artifacts

        # compute precision scores
        precision = self.get_metric(tf.keras.metrics.Precision(), test_labels, predictions_bin) # for both artifacts
        precision_musc = self.get_metric(tf.keras.metrics.Precision(), test_labels_musc, prediction_musc_bin) # only for muscle artifacts
        precision_eyem = self.get_metric(tf.keras.metrics.Precision(), test_labels_eyem, prediction_eyem_bin) # only for eye movement artifacts

        # compute recall scores
        recall = self.get_metric(tf.keras.metrics.Recall(), test_labels, predictions_bin)
        recall_musc = self.get_metric(tf.keras.metrics.Recall(), test_labels_musc, prediction_musc_bin)
        recall_eyem = self.get_metric(tf.keras.metrics.Recall(), test_labels_eyem, prediction_eyem_bin)

        # compute f1 scores
        f1 = self.get_f1(precision=precision, recall=recall)
        f1_musc = self.get_f1(precision=precision_musc, recall=recall_musc)
        f1_eyem = self.get_f1(precision=precision_eyem, recall=recall_eyem)

        print(f"Accuracy: {accuracy:.4f}, Muscle Accuracy: {accuracy_musc:.4f}, Eye Movement Accuracy: {accuracy_eyem:.4f}")
        print(f"Precision: {precision:.4f}, Muscle Precision: {precision_musc:.4f}, Eye Movement Precision: {precision_eyem:.4f}")
        print(f"Recall: {recall:.4f}, Muscle Recall: {recall_musc:.4f}, Eye Movement Recall: {recall_eyem:.4f}")
        print(f"F1-Score: {f1:.4f}, Muscle F1-Score: {f1_musc:.4f}, Eye Movement F1-Score: {f1_eyem:.4f}")
