from multiprocessing.dummy import Array
import torch
import torch.nn as nn
import numpy as np
import pre_data
array=[[[ 0.0108, -0.0091,  0.0614,  0.0500, -0.0512],
         [ 0.0032, -0.0330,  0.0605,  0.0568, -0.0297],
         [-0.0085, -0.0170,  0.0735,  0.0553, -0.0402],
         [-0.0049, -0.0136,  0.0780,  0.0549, -0.0579],
         [ 0.0079, -0.0266,  0.0782,  0.0445, -0.0361],
         [-0.0019, -0.0373,  0.0921,  0.0608, -0.0234],
         [-0.0049, -0.0136,  0.0780,  0.0549, -0.0579],
         [-0.0143, -0.0041,  0.0451,  0.0683, -0.0467],
         [ 0.0161, -0.0385,  0.0875,  0.0668, -0.0475],
         [-0.0181, -0.0240,  0.0662,  0.0611, -0.0480],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462],
         [-0.0018, -0.0136,  0.0765,  0.0810, -0.0462]]]

l=[[4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

dict={2:1,3:9,4:5}
print(list(dict.keys())[list(dict.values()).index(1)])