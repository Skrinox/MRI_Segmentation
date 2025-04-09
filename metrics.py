import numpy as np

def precision_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_pixel_pred = np.sum(pred_mask)

    if total_pixel_pred == 0:
        if np.sum(groundtruth_mask) == 0:
            return 1.0 # No predictions made; if ground truth is also empty, it's correct.
        else:
            return 0.0 # No predictions made; ground truth is non-empty, so it's incorrect.
    
    return round(intersect / total_pixel_pred, 3)

def recall_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    
    if total_pixel_truth == 0:
        return 1.0 if np.sum(pred_mask) == 0 else 0.0  # If prediction is also empty, it's correct.
    
    return round(intersect / total_pixel_truth, 3)

def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    
    if total_sum == 0:
        return 1.0  # Both are empty, perfect agreement.
    
    return round(2 * intersect / total_sum, 3)

def iou_score(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    
    if union == 0:
        return 1.0 if intersect == 0 else 0.0  # If both are empty, IoU is 1; if only prediction is non-zero, IoU is 0.
    
    return round(intersect / union, 3)


