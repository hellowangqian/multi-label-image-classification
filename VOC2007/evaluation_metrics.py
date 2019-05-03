"""
functions used to calculate the metrics for multi-label classification
cmap=mAP, emap=MiAP
"""
import numpy as np
import pdb
    
def cemap_cal(y_pred,y_true):
    '''
    y_true: -1 negative; 0 difficult_examples; 1 positive.
    '''
    nTest = y_true.shape[0]
    nLabel = y_true.shape[1]
    ap = np.zeros(nTest)
    for i in range(0,nTest):
        R = np.sum(y_true[i,:]==1)
        for j in range(0,nLabel):            
            if y_true[i,j]==1:
                r = np.sum(y_pred[i,np.nonzero(y_true[i,:]!=0)]>=y_pred[i,j])
                rb = np.sum(y_pred[i,np.nonzero(y_true[i,:]==1)] >= y_pred[i,j])

                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    emap = np.nanmean(ap)


    ap = np.zeros(nLabel)
    for i in range(0,nLabel):
        R = np.sum(y_true[:,i]==1)
        for j in range(0,nTest):
            if y_true[j,i]==1:
                r = np.sum(y_pred[np.nonzero(y_true[:,i]!=0),i] >= y_pred[j,i])
                rb = np.sum(y_pred[np.nonzero(y_true[:,i]==1),i] >= y_pred[j,i])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    cmap = np.nanmean(ap)

    return cmap,emap
def prf_cal(y_pred,y_true,k):
    """
    function to calculate top-k precision/recall/f1-score
    y_true: 0 1
    """
    GT=np.sum(y_true[y_true==1.])
    instance_num=y_true.shape[0]
    prediction_num=instance_num*k

    sort_indices = np.argsort(y_pred)
    sort_indices=sort_indices[:,::-1]
    static_indices = np.indices(sort_indices.shape)
    sorted_annotation= y_true[static_indices[0],sort_indices]
    top_k_annotation=sorted_annotation[:,0:k]
    TP=np.sum(top_k_annotation[top_k_annotation==1.])
    recall=TP/GT
    precision=TP/prediction_num
    f1=2.*recall*precision/(recall+precision)
    return precision, recall, f1

def cemap_cal_old(y_pred,y_true):
    """
    function to calculate C-MAP (mAP) and E-MAP
    y_true: 0 1
    """
    nTest = y_true.shape[0]
    nLabel = y_true.shape[1]
    ap = np.zeros(nTest)
    for i in range(0,nTest):
        R = np.sum(y_true[i,:])
        for j in range(0,nLabel):            
            if y_true[i,j]==1:
                r = np.sum(y_pred[i,:]>=y_pred[i,j])
                rb = np.sum(y_pred[i,np.nonzero(y_true[i,:])] >= y_pred[i,j])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    emap = np.nanmean(ap)

    ap = np.zeros(nLabel)
    for i in range(0,nLabel):
        R = np.sum(y_true[:,i])
        for j in range(0,nTest):
            if y_true[j,i]==1:
                r = np.sum(y_pred[:,i] >= y_pred[j,i])
                rb = np.sum(y_pred[np.nonzero(y_true[:,i]),i] >= y_pred[j,i])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    cmap = np.nanmean(ap)

    return cmap,emap
