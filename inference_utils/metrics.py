from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc



def get_accuracy(pred_label, gt_label):
    return accuracy_score(pred_label, gt_label)

def get_expected_calibration_error(y_pred, y_true, num_bins=15):

    #Code taken from https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html

    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0

    x,y = [], []
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))
            x.append(np.sum(prob_y[mask])/len(correct[mask]))
            y.append(np.sum(correct[mask])/len(correct[mask]))

    return o / y_pred.shape[0]
        
def get_auroc_score(unc_in, unc_out):
    in_labels = np.zeros(unc_in.shape)
    ood_labels = np.ones(unc_out.shape)
    return roc_auc_score(np.concatenate(in_labels, ood_labels), np.concatenate(unc_in, unc_out))
    


    
