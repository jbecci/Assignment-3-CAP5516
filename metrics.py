import numpy as np
from skimage.measure import label
from scipy.optimize import linear_sum_assignment 

def get_fast_aji(true, pred):
    """
    Compute AJI between ground truth and predicted instance masks
    """
    #convert to labeled instance masks
    true = label(true)
    pred = label(pred)

    true_ids = np.unique(true)[1:]  #remove background
    pred_ids = np.unique(pred)[1:]

    #instersection give number of pixels shared between true and predicted
    intersection = np.zeros((len(true_ids), len(pred_ids))) 

    #count overlapping pixels, populate matrix
    for i, t in enumerate(true_ids): #i for true 
        t_mask = (true == t)
        for j, p in enumerate(pred_ids): #j for predicted
            p_mask = (pred == p)
            intersection[i, j] = np.sum(t_mask & p_mask)

    union = np.zeros_like(intersection) #store pixels from true OR predicted
    for i, t in enumerate(true_ids):
        t_mask = (true == t)
        for j, p in enumerate(pred_ids):
            p_mask = (pred == p)
            union[i, j] = np.sum(t_mask | p_mask)

    #initialize AJI
    aji = 0
    matched_true = set() #keep track of match
    matched_pred = set()
    for _ in range(min(len(true_ids), len(pred_ids))):
        i, j = np.unravel_index(intersection.argmax(), intersection.shape)
        if intersection[i, j] == 0:
            break
        aji += intersection[i, j] / union[i, j]
        matched_true.add(i)
        matched_pred.add(j)
        intersection[i, :] = 0
        intersection[:, j] = 0

    unmatched_true = set(range(len(true_ids))) - matched_true #keep track of unmatched
    unmatched_pred = set(range(len(pred_ids))) - matched_pred

    #combine match and unmatched to get final AJI score
    total = sum([np.sum(true == true_ids[i]) for i in unmatched_true]) + \
            sum([np.sum(pred == pred_ids[j]) for j in unmatched_pred]) + \
            aji * len(matched_true)
    return aji * len(matched_true) / total if total > 0 else 0 #final score

def get_pq(true, pred, iou_thresh=0.5):
    """
    Compute PQ between ground truth and predicted instance masks
    """
    true = label(true)
    pred = label(pred)

    true_ids = np.unique(true)[1:]
    pred_ids = np.unique(pred)[1:]

    #find iou for all pairs or true/predicted instance
    iou_matrix = np.zeros((len(true_ids), len(pred_ids)))

    for i, t_id in enumerate(true_ids):
        t_mask = (true == t_id)
        for j, p_id in enumerate(pred_ids):
            p_mask = (pred == p_id)
            inter = np.sum(t_mask & p_mask)
            union = np.sum(t_mask | p_mask)
            if union == 0:
                iou_matrix[i, j] = 0
            else:
                iou_matrix[i, j] = inter / union

    matched = iou_matrix > iou_thresh #check if iou exceeds threshold
    tp = np.sum(matched) #true positives
    fp = len(pred_ids) - np.sum(matched.any(axis=0)) #false positives
    fn = len(true_ids) - np.sum(matched.any(axis=1)) #false negatives

    #final pq score use panoptic quality formula
    pq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    return pq
