import numpy as np


def regular_info_gain(y_main, left, right, weights, criterion):
    heuristic_left  = 0 if len(left) == 0 else criterion(left)
    heuristic_right = 0 if len(right) == 0 else criterion(right)
    heuristic_all = criterion(y_main)
    val = heuristic_all - (len(left)/len(y_main)*heuristic_left + len(right)/len(y_main)*heuristic_right)
    return val

def ImportanceWeighing(y_main, left, right, weight, criterion):
    heuristic_left  = 0 if len(left) == 0 else criterion(left)
    heuristic_right = 0 if len(right) == 0 else criterion(right)
    heuristic_all = criterion(y_main)
    val = heuristic_all - (len(left)/len(y_main)*heuristic_left + len(right)/len(y_main)*heuristic_right)
    val = weight * val
    return val

def ImportanceMinimization(y_main, left, right, weight, criterion):
    heuristic_left  = 0 if len(left) == 0 else criterion(left)
    heuristic_right = 0 if len(right) == 0 else criterion(right)
    val = weight[0]*len(left)/len(y_main)*heuristic_left + weight[1]*len(right)/len(y_main)*heuristic_right
    return val