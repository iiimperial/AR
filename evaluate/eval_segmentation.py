import argparse
import os
import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff

def dice_coeff(pred, gt):
    tp = np.sum(pred*gt)
    fp = np.sum((1-gt)*pred)
    fn = np.sum((1-pred)*gt)
    return 2*tp/(2*tp+fp+fn +1e-6)

def jaccard(pred,gt):
    tp = np.sum(pred*gt)
    fp = np.sum((1-gt)*pred)
    fn = np.sum((1-pred)*gt)
    return tp/(tp+fp+fn+1e-6)

def hausdorff_distance(pred,gt):
    predpts = np.argwhere(pred>0)
    gtpts = np.argwhere(gt>0)
    if len(predpts)==0 or len(gtpts)==0:
        return np.inf
    d1 = directed_hausdorff(predpts, gtpts)[0]
    d2 = directed_hausdorff(gtpts, predpts)[0]
    return max(d1,d2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir",type=str,required=True)
    parser.add_argument("--gt_dir",type=str,required=True)
    args = parser.parse_args()
    pred_files = sorted(os.listdir(args.pred_dir))
    dice_list = []
    jac_list = []
    hd_list = []
    for fn in pred_files:
        ppath = os.path.join(args.pred_dir, fn)
        gpath = os.path.join(args.gt_dir, fn)
        pred = (cv2.imread(ppath,0)>127).astype(np.float32)
        gt = (cv2.imread(gpath,0)>127).astype(np.float32)
        dice_list.append(dice_coeff(pred,gt))
        jac_list.append(jaccard(pred,gt))
        hd_list.append(hausdorff_distance(pred,gt))
    print(f"mean Dice: {np.mean(dice_list):.3f} ± {np.std(dice_list):.3f}")
    print(f"mean Jaccard: {np.mean(jac_list):.3f} ± {np.std(jac_list):.3f}")
    print(f"mean Hausdorff: {np.mean(hd_list):.3f} ± {np.std(hd_list):.3f}")

if __name__ == "__main__":
    main()