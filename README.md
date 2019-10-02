# mcv-m1-code

Master Computer Vision, Module M1: Distributed code


## test_submission.py: 
- This script allows to test your submission. FOr each query set, you can submit up to two different methods. You must submit your results to:

```
/home/dlcv0X/m1-results/weekY/qstW/methodZ
```

where X is the team number, Y is the week number (1 to 5), W is the QS number (1 or 2) and Z is the method number (1 or 2). 
Example of usage of the submission script (for week1, Team 1):

```
python test_submission.py --help  # How to use 
python test_submission.py 1 1 0 1 # Test submission for QST1
python test_submission.py 1 1 0 2 # Test submission for QST2
```

The script also shows how to compute the MAP@k metric and to compute Precision, Recall and F1 given GT and hypothesis masks.

## virtualenvs.txt
- Info to create a virtualenv to run this code.


## utils/
- Folder with some useful scripts
