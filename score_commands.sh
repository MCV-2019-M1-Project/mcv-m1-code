#!/bin/bash

k_val=5
week=5
ii=1    # Team number

touch qst1_w${week}_score${k_val}

#textboxes1="--gtTextBoxes /home/dlcv/2019/Query/qst1_w${week}/text_boxes.pkl"
#textboxes2="--gtTextBoxes /home/dlcv/2019/Query/qst2_w${week}/text_boxes.pkl"
textboxes1=""
textboxes2=""

exedir='/home/dlcv/2019/m1-project-paintings/'

name="/home/dlcv0${ii}/m1-results/week${week}/QST1"
python ${exedir}scoring/score_painting_retrieval_new.py ${name} ../Query/qst1_w${week}/gt_corresps.pkl --imaDir ../Query/qst1_w${week} ${textboxes1} --kVal=${k_val} --gtPicFrames ../Query/qst1_w${week}/frames.pkl >> qst1_w${week}_score${k_val}

