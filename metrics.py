import numpy as np
from torch import Tensor
from sklearn.metrics import roc_curve, auc, hamming_loss, accuracy_score
import pdb

CLASSIFICATION_THRESHOLD: float = 0.5  # Best keep it in [0.0, 1.0] range

# def accuracy(out, labels):
#     outputs = np.argmax(out, axis=1)
#     return np.sum(outputs == labels)

def onehot(index_array: np.ndarray, nb_classes: int, dtype=np.float32):
    return np.eye(nb_classes, dtype=dtype)[index_array]


def confusion_per_class(predictions: np.ndarray, labels: np.ndarray, nb_classes: int):
    predictions = onehot(predictions, nb_classes, np.bool8)
    labels = onehot(labels, nb_classes, np.bool8)

    n_predictions = ~predictions
    n_labels = ~labels

    tp_per_class = (predictions & labels).sum(0).astype(np.float32)
    fp_per_class = (predictions & n_labels).sum(0).astype(np.float32)
    fn_per_class = (n_predictions & labels).sum(0).astype(np.float32)
    return tp_per_class, fp_per_class, fn_per_class


def micro_f1_score(tp_per_class, fp_per_class, fn_per_class):
    total_tp = tp_per_class.sum()
    total_fp = fp_per_class.sum()
    total_fn = fn_per_class.sum()
    del tp_per_class
    del fp_per_class
    del fn_per_class

    total_precision = total_tp / (total_tp + total_fp + 1e-12)
    total_recall = total_tp / (total_tp + total_fn + 1e-12)

    micro_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-12)
    del total_precision
    del total_recall

    return micro_f1


def www_macro_f1_score(tp_per_class: np.ndarray, fp_per_class: np.ndarray, fn_per_class: np.ndarray):
    is_nonzero_prediction = (tp_per_class + fp_per_class) != 0
    is_nonzero_actual = (tp_per_class + fn_per_class) != 0

    where_nonzero_prediction = np.squeeze(is_nonzero_prediction.nonzero(), axis=-1)
    where_nonzero_actual = np.squeeze(is_nonzero_actual.nonzero(), axis=-1)
    del is_nonzero_prediction, is_nonzero_actual

    precision_per_class = tp_per_class[where_nonzero_prediction] / (
                tp_per_class[where_nonzero_prediction] + fp_per_class[where_nonzero_prediction])
    recall_per_class = tp_per_class[where_nonzero_actual] / (tp_per_class[where_nonzero_actual] + fn_per_class[where_nonzero_actual])
    del tp_per_class, fp_per_class, fn_per_class

    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    del precision_per_class, recall_per_class

    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
    return macro_f1


def compute_metrics(preds, labels, nb_classes):
    assert len(preds) == len(labels)

    tp_per_class, fp_per_class, fn_per_class = confusion_per_class(preds, labels, nb_classes)
    micro_f1 = micro_f1_score(tp_per_class, fp_per_class, fn_per_class)
    # macro_f1 = www_macro_f1_score(tp_per_class, fp_per_class, fn_per_class)
    macro_f1 = 0
    return {"micro_f1": micro_f1, "macro_f1": macro_f1}





def accuracy(y_pred: Tensor, y_true: Tensor):

    # inference code
    # output_odp = []
    # for arr in y_pred:
    #     t = (-arr).argsort()[:5]
    #     output_odp.append(t.tolist())
    # import pickle
    # file_path = 'D:/바탕화면/(논문)multi-pretraining/NYT'
    # with open(file_path+'/xlnet_top5','wb') as f:
    #     pickle.dump(output_odp,f)
    # print('123123',y_pred, y_pred.size())
    # print('preds :', output_odp, 'labels :', labels)
    outputs = np.argmax(y_pred, axis=1)
    preds = outputs.numpy()
    labels = y_true.detach().cpu().numpy()
    result = compute_metrics(preds, labels, nb_classes=2531)
    print(result)
    return np.mean(outputs.numpy() == y_true.detach().cpu().numpy())


def accuracy_multilabel(y_pred: Tensor, y_true: Tensor, sigmoid: bool = True):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    outputs = np.argmax(y_pred, axis=1)
    real_vals = np.argmax(y_true, axis=1)
    return np.mean(outputs.numpy() == real_vals.numpy())


def accuracy_thresh(y_pred: Tensor, y_true: Tensor, thresh: float = CLASSIFICATION_THRESHOLD, sigmoid: bool = True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return ((y_pred > thresh) == y_true.byte()).float().mean().item()
#     return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_pred: Tensor, y_true: Tensor, thresh: float = 0.3, beta: float = 2, eps: float = 1e-9, sigmoid: bool = True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean().item()


def roc_auc(y_pred: Tensor, y_true: Tensor):
    # ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc["micro"]


def Hamming_loss(y_pred: Tensor, y_true: Tensor, sigmoid: bool = True, thresh: float = CLASSIFICATION_THRESHOLD, sample_weight=None):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    return hamming_loss(y_true, y_pred, sample_weight=sample_weight)


def Exact_Match_Ratio(y_pred: Tensor, y_true: Tensor, sigmoid: bool = True, thresh: float = CLASSIFICATION_THRESHOLD, normalize: bool = True, sample_weight=None):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    return accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)


def F1(y_pred: Tensor, y_true: Tensor, threshold: float = CLASSIFICATION_THRESHOLD):
    return fbeta(y_pred, y_true, thresh=threshold, beta=1)
