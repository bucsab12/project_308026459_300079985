import numpy as np
from sklearn.metrics import confusion_matrix
import sys


def calc_metrics(true_labels, pred_labels, modes=['class', 'TC', 'WT', 'EN']):
    """
    This function calculates the metrics between two segmentation masks.
    :param true_labels: Ground Truth segmentation mask.
    :param pred_labels: Predicted segmentation mask.
    :param modes: Modes for evaluation between the two segmentation masks.
    class - calculates the metrics for each label type: 1 - Necrotic tissue, 2 - Edema tissue, 3 - Enhanced Tumor, 0 - Everything else
    TC (Tumor Core) - calculates the metrics for the tumor core region (labels 1 and 3 are joined).
    WT (Whole Tumor) -  calculates the metrics for the Whole Tumor (labels 1, 2 and 3 are joined).
    EN (Enhanced) - calculates the metrics for the enhanced tumor Tumor (label 3).
    :return: None
    """
    for mode in modes:
        true_labels_cp = np.copy(true_labels)
        pred_labels_cp = np.copy(pred_labels)
        if mode.lower() == 'class':
            classes = np.arange(4)
            class_list_str = ['Non-tumor/Background', 'Necrotic Tissue', 'Edema Tissue', 'Enhancing Tumor']
        elif mode.lower() == 'tc':
            classes = np.arange(3)
            true_labels_cp[true_labels_cp == 3] = 1
            pred_labels_cp[pred_labels_cp == 3] = 1
            class_list_str = ['Non-tumor/Background', 'Tumor Core', 'Edema Tissue']
        elif mode.lower() == 'wt':
            classes = np.arange(2)
            true_labels_cp[true_labels_cp == 2] = 1
            true_labels_cp[true_labels_cp == 3] = 1
            pred_labels_cp[pred_labels_cp == 2] = 1
            pred_labels_cp[pred_labels_cp == 3] = 1
            class_list_str = ['Non-tumor/Background', 'Whole Tumor']
        elif mode.lower() == 'en':
            classes = np.arange(2)
            true_labels_cp[true_labels_cp == 1] = 0
            true_labels_cp[true_labels_cp == 2] = 0
            true_labels_cp[true_labels_cp == 3] = 1
            pred_labels_cp[pred_labels_cp == 1] = 0
            pred_labels_cp[pred_labels_cp == 2] = 0
            pred_labels_cp[pred_labels_cp == 3] = 1
            class_list_str = ['Non-tumor/Background', 'Enhanced']
        else:
            print('Modes {} is not an allowed mode.'.format(mode))
            print('Allowed modes are "class", "TC" (Tumor Core) and "WT" (whole tumor).')
            print('Please select one of the specified modes and run the script again.')
            sys.exit(0)
        fp_dict, fn_dict, tp_dict, tn_dict = {}, {}, {}, {}
        # set default value of 0 for all metrics across all dictionaries
        for dict_name in [fp_dict, fn_dict, tp_dict, tn_dict]:
            for class_id in range(len(classes)):
                dict_name.setdefault(class_id, 0)
        # Looping over all the slices and calculating the metrics
        cm = confusion_matrix(true_labels_cp, pred_labels_cp, labels=classes)
        for class_id in range(len(classes)):
            fp, fn, tp, tn = calculate_metrics(cm, class_id)
            fp_dict[class_id] += fp
            fn_dict[class_id] += fn
            tp_dict[class_id] += tp
            tn_dict[class_id] += tn
        print_metrics(fp_dict, fn_dict, tp_dict, tn_dict, class_list_str)


def calculate_metrics(confusion_matrix_ins, class_id):
    """
    Funtion that calculates the TP, FP, TN, FN values between the ground truth and predicted masks from a given confusion matrix
    :param confusion_matrix_ins: The confusion matrix from which to calculate the TP, FP, TN, FN
    :param class_id: Index of relevant class
    :return: TP, FP, TN, FN values
    """
    fp = confusion_matrix_ins[:, class_id].sum() - np.diag(confusion_matrix_ins)[class_id]
    fn = confusion_matrix_ins[class_id, :].sum() - np.diag(confusion_matrix_ins)[class_id]
    tp = np.diag(confusion_matrix_ins)[class_id]
    tn = confusion_matrix_ins.sum() - (fp + fn + tp)
    return fp, fn, tp, tn


def print_metrics(fp_dict, fn_dict, tp_dict, tn_dict, class_list_str):
    """
    This function is used to print out an assortment of evaluation metrics between the ground truth and predicted masks
    """
    for class_id in range(len(fp_dict.keys())):
        fp = fp_dict[class_id]
        fn = fn_dict[class_id]
        tp = tp_dict[class_id]
        tn = tn_dict[class_id]
        print('Metrics for class {} - {}'.format(class_id, class_list_str[class_id]))
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = tp / (tp + fn)
        print('sensitivity: {}'.format(TPR))
        # Specificity or true negative rate
        TNR = tn / (tn + fp)
        print('Specificity: {}'.format(TNR))
        # Precision or positive predictive value
        PPV = tp / (tp + fp)
        print('Precision: {}'.format(PPV))
        # Negative predictive value
        NPV = tn / (tn + fn)
        print('Negative predictive value: {}'.format(NPV))
        # Fall out or false positive rate
        FPR = fp / (fp + tn)
        print('False Positive Rate: {}'.format(FPR))
        # False negative rate
        FNR = fn / (tp + fn)
        print('False Negative Rate: {}'.format(FNR))
        # False discovery rate
        FDR = fp / (tp + fp)
        print('False Discovery Rate: {}'.format(FDR))
        # Dice Score
        DSC = 2*tp / (2*tp + fp + fn)
        print('Dice score: {}'.format(DSC))
        print('\n')


# The path to the compressed numpy array that was saved during model evaluation over the validation data.
PATH = 'VALIDATION/val_results.npz'
y_true = np.argmax(np.load(PATH)['a'], axis=-1)
y_pred = np.argmax(np.load(PATH)['b'], axis=-1)

# 0 for everything else
# 1 for necrotic
# 2 for edema
# 3 for enhancing tumor (we changed the label from 4 to 3 during execution)

calc_metrics(y_true, y_pred, modes=['class', 'TC', 'WT', 'EN'])
