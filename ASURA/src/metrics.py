import cv2
import numpy as np

import util


def jaccard(img, gt):
    inter = np.sum(np.logical_and(img > 0.5, gt > 0.5))
    union = np.sum(np.logical_or(img > 0.5, gt > 0.5))
    return float(inter)/union


def dice_coeff(img, gt) :
    inter = np.sum(np.logical_and(img > 0.5, gt > 0.5))
    s_img = np.sum(img > 0.5)
    s_gt = np.sum(gt > 0.5)
    return (2.0*inter)/(s_img+s_gt)


def precision_recall_f(img, gt) :
    tp = np.sum(np.logical_and(img > 0.5, gt > 0.5))
    fp = np.sum(np.logical_and(img > 0.5, gt < 0.5))
    tn = np.sum(np.logical_and(img < 0.5, gt < 0.5))
    fn = np.sum(np.logical_and(img < 0.5, gt > 0.5))
    
    acc = float(tp+tn)/(tp+tn+fp+fn)
    
    if tp+fp == 0 :
        prec = 0
    else :
        prec = float(tp)/(tp+fp)
    
    if tp+fn == 0 :
        recall = 0
    else :
        recall = float(tp)/(tp+fn)
    
    if prec == 0 and recall == 0 :
        f_meas = 0.0
    else :
        f_meas = 2*prec*recall/(prec+recall)

    return acc, prec, recall, f_meas   


def calculate_metrics(img, gt, index= 1) :
    new_img = util.treat_mask(img)[:,:,index]
    new_gt = util.treat_mask(gt)[:,:,index]
    j = jaccard(new_img, new_gt)
    d = dice_coeff(new_img, new_gt)
    a,p,r,f = precision_recall_f(new_img, new_gt)

    return np.array([j, d, a, p, r, f])

