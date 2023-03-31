import sys
sys.path.append('./')
from mylibs import *

class Tester:
    
    def load_model(self, logdir, ckptdir=None):
        model = keras.models.load_model(logdir)
        if ckptdir!=None:
            model.load_weights(ckptdir)
        return model
    
    def get_metric(self, metric, label, pred):
        m = metric
        m.update_state(label, pred)
        result = m.result().numpy()
        m.reset_state()
        return result
    
    def get_f1(self, precision, recall):
        return 2*precision*recall*(precision+recall)
    
    def get_metrics(self, model, test_features, test_labels):
        predictions = model.predict(test_features)
        
        prediction_musc = np.squeeze(predictions[:,:,0])
        prediction_eyem = np.squeeze(predictions[:,:,1])
        test_labels_musc = np.squeeze(test_labels[:,:,0])
        test_labels_eyem = np.squeeze(test_labels[:,:,1])

        # Convert the predicted probabilities to binary labels
        predictions_bin = tf.cast(tf.greater(predictions, 0.5), tf.int32)
        prediction_musc_bin = tf.cast(tf.greater(prediction_musc, 0.5), tf.int32)
        prediction_eyem_bin = tf.cast(tf.greater(prediction_eyem, 0.5), tf.int32)


        accuracy = self.get_metric(tf.keras.metrics.BinaryAccuracy(), test_labels, predictions_bin)
        accuracy_musc = self.get_metric(tf.keras.metrics.BinaryAccuracy(), test_labels_musc, prediction_musc_bin)
        accuracy_eyem = self.get_metric(tf.keras.metrics.BinaryAccuracy(), test_labels_eyem, prediction_eyem_bin)

        precision = self.get_metric(tf.keras.metrics.Precision(), test_labels, predictions_bin)
        precision_musc = self.get_metric(tf.keras.metrics.Precision(), test_labels_musc, prediction_musc_bin)
        precision_eyem = self.get_metric(tf.keras.metrics.Precision(), test_labels_eyem, prediction_eyem_bin)

        recall = self.get_metric(tf.keras.metrics.Recall(), test_labels, predictions_bin)
        recall_musc = self.get_metric(tf.keras.metrics.Recall(), test_labels_musc, prediction_musc_bin)
        recall_eyem = self.get_metric(tf.keras.metrics.Recall(), test_labels_eyem, prediction_eyem_bin)

        f1 = self.get_f1(precision=precision, recall=recall)
        f1_musc = self.get_f1(precision=precision_musc, recall=recall_musc)
        f1_eyem = self.get_f1(precision=precision_eyem, recall=recall_eyem)

        print(f"Accuracy: {accuracy:.4f}, Muscle Accuracy: {accuracy_musc:.4f}, Eye Movement Accuracy: {accuracy_eyem:.4f}")
        print(f"Precision: {precision:.4f}, Muscle Precision: {precision_musc:.4f}, Eye Movement Precision: {precision_eyem:.4f}")
        print(f"Recall: {recall:.4f}, Muscle Recall: {recall_musc:.4f}, Eye Movement Recall: {recall_eyem:.4f}")
        print(f"F1-Score: {f1:.4f}, Muscle F1-Score: {f1_musc:.4f}, Eye Movement F1-Score: {f1_eyem:.4f}")
