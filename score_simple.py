"""
Score painting retrieval tasks

Usage:
  score_painting_retrieval.py <inputDir> <gtQueryFile> [--gtTextBoxes=<gtb>] [--imaDir=<id>] [--kVal=<kv>] [--outDir=<od>]
  score_painting_retrieval.py -h | --help
Options:
  --gtTextBoxes=<gtb>      Ground truth for text boxes (.pkl) [default: None]
  --imaDir=<id>            Images dir (necessary to evaluate bboxes) (.pkl) [default: None]
  --kVal=<kv>              Value of k for map@k computation [default: 3]
  --outDir=<od>            Directory for output masks [default: None]
"""

import numpy as np
import cv2
import fnmatch
import os
import sys
import pickle
from docopt import docopt
import imageio
from ml_metrics import mapk,apk
from sklearn.metrics import jaccard_similarity_score
import evaluation.evaluation_funcs as evalf
import geometry_utils as gu
from operator import itemgetter 


# Compute the depth of a list (of lists (of lists ...) ...)
# Empty list not allowed!!
#https://stackoverflow.com/questions/6039103/counting-depth-or-the-deepest-level-a-nested-list-goes-to
list_depth = lambda L: isinstance(L, list) and max(map(list_depth, L))+1

def add_list_level(input_list):
    out = []
    for ll in input_list:
        tmp = []
        for q in ll:
            tmp.append([q])
        out.append(tmp)
    return (out)

def compute_mapk(gt,hypo,k_val):

    hypo = list(hypo)
    if list_depth(hypo) == 2:
        hypo = add_list_level(hypo.copy())

    apk_list = []
    for ii,query in enumerate(gt):
        for jj,sq in enumerate(query):
            apk_val = 0.0
            if len(hypo[ii]) > jj:
                apk_val = apk([sq],hypo[ii][jj], k_val)
            apk_list.append(apk_val)
            
    return np.mean(apk_list)

def score_bboxes(hypo_bboxes, gt_bboxes):

    if len(hypo_bboxes) != len(gt_bboxes):
        print ('Error: the number of bboxes in GT and hypothesis differ!')
        sys.exit()
    
    bboxTP = 0
    bboxFN = 0
    bboxFP = 0

    bbox_precision = 0
    bbox_accuracy  = 0

    iou = 0
    for ii in range(len(gt_bboxes)):
        [local_bboxTP, local_bboxFN, local_bboxFP,local_iou] = evalf.performance_accumulation_window(hypo_bboxes[ii], gt_bboxes[ii])
        bboxTP = bboxTP + local_bboxTP
        bboxFN = bboxFN + local_bboxFN
        bboxFP = bboxFP + local_bboxFP
        iou    = iou + local_iou

    # Plot performance evaluation
    [bbox_precision, bbox_sensitivity, bbox_accuracy] = evalf.performance_evaluation_window(bboxTP, bboxFN, bboxFP)
    iou = iou / bboxTP
    
    bboxF1 = 0
    if (bbox_precision + bbox_sensitivity) != 0:
        bbox_f1 = 2*((bbox_precision*bbox_sensitivity)/(bbox_precision + bbox_sensitivity))

    return bbox_precision, bbox_sensitivity, bbox_f1, bbox_accuracy, iou
        

def score_pixel_masks(result_masks, test_masks):
    pixelTP  = 0
    pixelFN  = 0
    pixelFP  = 0
    pixelTN  = 0


    for ii, mask_name in enumerate(result_masks):

        # Read mask file
        pixelCandidates = imageio.imread(mask_name)>0
        if len(pixelCandidates.shape) == 3:
            pixelCandidates = pixelCandidates[:,:,0]

        # Accumulate pixel performance of the current image %%%%%%%%%%%%%%%%%
        gt_mask_name = test_masks[ii]

        pixelAnnotation = imageio.imread(gt_mask_name)>0
        if len(pixelAnnotation.shape) == 3:
            pixelAnnotation = pixelAnnotation[:,:,0]


        if pixelAnnotation.shape != pixelCandidates.shape:
            print ('Error: hypothesis ({}) and  GT masks ({})dimensions do not match!'.format(pixelCandidates.shape,pixelAnnotation.shape))
            sys.exit()

        [localPixelTP, localPixelFP, localPixelFN, localPixelTN] = evalf.performance_accumulation_pixel(pixelCandidates, pixelAnnotation)
        pixelTP = pixelTP + localPixelTP
        pixelFP = pixelFP + localPixelFP
        pixelFN = pixelFN + localPixelFN
        pixelTN = pixelTN + localPixelTN

    [pixelPrecision, pixelAccuracy, pixelSpecificity, pixelSensitivity] = evalf.performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
    pixelF1 = 0
    if (pixelPrecision + pixelSensitivity) != 0:
        pixelF1 = 2*((pixelPrecision*pixelSensitivity)/(pixelPrecision + pixelSensitivity))
        
    return pixelPrecision, pixelSensitivity, pixelF1      

    
          
if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
     
    input_dir     = args['<inputDir>']
    gt_query_file = args['<gtQueryFile>']
    
    gt_text_boxes = args['--gtTextBoxes']
    ima_dir       = args['--imaDir']
    k_val         = int(args['--kVal'])
    out_dir       = args['--outDir']
    
    print ('inputDir        = {}'.format(input_dir), file=sys.stderr)
    print ('gtQueryFile     = {}'.format(gt_query_file), file=sys.stderr)
    
    print ('gtTextBoxes     = {}'.format(gt_text_boxes), file=sys.stderr)
    print ('imaDir          = {}'.format(ima_dir), file=sys.stderr)
    print ('kVal            = {}'.format(k_val), file=sys.stderr)
    print ('outDir          = {}'.format(out_dir), file=sys.stderr)

    # Query GT. Must be always present
    with open(gt_query_file, 'rb') as gtfd:
        gtquery_list = pickle.load(gtfd)

    if gt_text_boxes != 'None':
        with open(gt_text_boxes, 'rb') as gttb:
            gt_tboxes = pickle.load(gttb)

        
    # Use full path to results in students home (i.e. /home/mcv02/m1-results/week5/QST1')        
    path_list = os.path.normpath(input_dir).split(os.path.sep)
    team = path_list[2]
    week = path_list[4]

    images_list = sorted(fnmatch.filter(os.listdir(ima_dir), '*.png'))
    images_list = ['{}/{}'.format(ima_dir,xx) for xx in images_list]

    for dirName, subdirList, fileList in os.walk(input_dir):
        for fname in fileList:
            if fname == 'result.pkl': 
                method = os.path.normpath(dirName).split(os.path.sep)[-1]
                hypo_name = '{}/{}'.format(dirName, fname)
                masks_list = sorted(fnmatch.filter(os.listdir(dirName), '*.png'))
                masks_list = ['{}/{}'.format(dirName,xx) for xx in masks_list]
                
                with open (hypo_name, 'rb') as fd:
                    hypo = pickle.load(fd)

                score = compute_mapk(gtquery_list, hypo, k_val)
                print ('Q,{}, {}, {:.3f}'.format(team, method, score))

                if len(masks_list) == len(gtquery_list):
                    pixelPrecision, pixelSensitivity, pixelF1 = score_pixel_masks(masks_list, images_list)
                    #print ('Team {} background, method {} : Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}\n'.format(team, method, pixelPrecision, pixelSensitivity, pixelF1))      
                    print ('PM,{},{},{:.2f}, {:.2f},{:.2f}'.format(team, method, pixelPrecision, pixelSensitivity, pixelF1))      
                    
            # Evaluate text bbox using bbox-based measures
            if fname == 'text_boxes.pkl' and gt_text_boxes != 'None':
                method = os.path.normpath(dirName).split(os.path.sep)[-1]
                text_hypo_name = '{}/{}'.format(dirName, fname)
                with open (text_hypo_name, 'rb') as fd:
                    hypo_tboxes = pickle.load(fd)
                
                bbox_precision, bbox_recall, bbox_f1, _, iou = score_bboxes(hypo_tboxes, gt_tboxes)
                print ('TB,{},{},{:.2f}, {:.2f},{:.2f},{:.2f}'.format(team, method, bbox_precision, bbox_recall, bbox_f1, iou))      

