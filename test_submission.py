#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Use this script to test your submission. Do not look at the results, as they are computed with fake annotations and masks.
This is just to see if there's any problem with files, paths, permissions, etc.
If you find a bug, please report it to ramon.morros@upc.edu 

Usage:
  test_submission.py <weekNumber> <teamNumber> <winEval> <querySet> [--testDir=<td>] 
  test_submission.py -h | --help
Options:
  --testDir=<td>        Directory with the test images & masks [default: /home/dlcv/DataSet/fake_test]
"""



import fnmatch
import os
import sys
import pickle
import imageio
from docopt import docopt
from ml_metrics import mapk
from sklearn.metrics import jaccard_similarity_score
#from evaluation.load_annotations import load_annotations
import evaluation.evaluation_funcs as evalf


if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    week      = int(args['<weekNumber>'])
    team      = int(args['<teamNumber>'])
    query_set = int(args['<querySet>'])
    
    window_evaluation = int(args['<winEval>'])  # Whether to perform or not window based evaluation: 0 for week 1, 1 for week 2

    # This folder contains fake masks and text annotations. Do not change this.
    test_dir = args['--testDir']

    k_val = 10

    
    # This folder contains your results: mask imaged and window list pkl files. Do not change this.
    results_dir = '/home/dlcv{:02d}/m1-results/week{}/QST{}'.format(team, week, query_set)

    # Load mask names in the given directory
    test_masks     = sorted(fnmatch.filter(os.listdir(test_dir), '*.png'))
    test_masks_num = len(test_masks)

    # Query GT. Must be always present
    gtquery_list = []
    gt_query_file = '{}/gt_corresps.pkl'.format(test_dir)
    with open(gt_query_file, 'rb') as gtfd:
        gtquery_list = pickle.load(gtfd)



    print (results_dir)


    # List all folders (corresponding to the different methods) in the results directory
    methods = next(os.walk(results_dir))[1]


    for method in methods:

        pixelTP  = 0
        pixelFN  = 0
        pixelFP  = 0
        pixelTN  = 0
        windowTP = 0
        windowFN = 0
        windowFP = 0 

        #print ('Method: {}\n'.format(method), file=sys.stderr)

        # Read masks (if any)
        result_masks     = sorted(fnmatch.filter(os.listdir('{}/{}'.format(results_dir, method)), '*.png'))
        result_masks_num = len(result_masks)

        # Correspondences Hypotesis file
        hypo_name = '{}/{}/result.pkl'.format(results_dir, method)
        with open(hypo_name, 'rb') as fd:
            hypo = pickle.load(fd)

        score = mapk(gtquery_list, hypo, k_val)

        print ('Team {}, method: {}, map@10: {:.3f}'.format(team, method, score))

        
        if result_masks_num != test_masks_num:
            print ('Method {} : {} result files found but there are {} test files'.format(method, result_masks_num, test_masks_num), file = sys.stderr) 


        for ii in range(len(result_masks)):

            # Read mask file
            candidate_masks_name = '{}/{}/{}'.format(results_dir, method, result_masks[ii])
            #print ('File: {}'.format(candidate_masks_name), file = sys.stderr)

            pixelCandidates = imageio.imread(candidate_masks_name)>0
            if len(pixelCandidates.shape) == 3:
                pixelCandidates = pixelCandidates[:,:,0]
            
            # Accumulate pixel performance of the current image %%%%%%%%%%%%%%%%%
            name, ext = os.path.splitext(test_masks[ii])
            gt_mask_name = '{}/{}.png'.format(test_dir, name)

            pixelAnnotation = imageio.imread(gt_mask_name)>0
            if len(pixelAnnotation.shape) == 3:
                pixelAnnotation = pixelAnnotation[:,:,0]


            if pixelAnnotation.shape != pixelCandidates.shape:
                print ('Error: hypothesis and  GT masks do not match!')
                sys.exit()

            [localPixelTP, localPixelFP, localPixelFN, localPixelTN] = evalf.performance_accumulation_pixel(pixelCandidates, pixelAnnotation)
            pixelTP = pixelTP + localPixelTP
            pixelFP = pixelFP + localPixelFP
            pixelFN = pixelFN + localPixelFN
            pixelTN = pixelTN + localPixelTN

            if window_evaluation == 1:
                # Read .pkl file
            
                name_r, ext_r = os.path.splitext(result_masks[ii])
                pkl_name      = '{}/{}/{}.pkl'.format(results_dir, method, name_r)
                

                with open(pkl_name, "rb") as fp:   # Unpickling
                    windowCandidates = pickle.load(fp)

                gt_annotations_name = '{}/gt/gt.{}.txt'.format(test_dir, name)
                windowAnnotations = load_annotations(gt_annotations_name)

                [localWindowTP, localWindowFN, localWindowFP] = evalf.performance_accumulation_window(windowCandidates, windowAnnotations)
                windowTP = windowTP + localWindowTP
                windowFN = windowFN + localWindowFN
                windowFP = windowFP + localWindowFP

        # Plot performance evaluation
        [pixelPrecision, pixelAccuracy, pixelSpecificity, pixelSensitivity] = evalf.performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
        pixelF1 = 0
        if (pixelPrecision + pixelSensitivity) != 0:
            pixelF1 = 2*((pixelPrecision*pixelSensitivity)/(pixelPrecision + pixelSensitivity))
        
        print ('Team {:02d} background, method {} : Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}\n'.format(team, method, pixelPrecision, pixelSensitivity, pixelF1))      

        if window_evaluation == 1:
            [windowPrecision, windowSensitivity, windowAccuracy] = evalf.performance_evaluation_window(windowTP, windowFN, windowFP) # (Needed after Week 3)
            windowF1 = 0
            if (windowPrecision + windowSensitivity) != 0:
                windowF1 = 2*((windowPrecision*windowSensitivity)/(windowPrecision + windowSensitivity))

            print ('Team {:02d} window, method {} : {:.2f}, {:.2f}, {:.2f}\n'.format(team, method, windowPrecision, windowSensitivity, windowF1)) 
