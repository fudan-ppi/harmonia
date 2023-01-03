#!/usr/bin/python3

import os 
from sys import argv

fileName = argv[1]
print(fileName)
file1 = open(fileName, 'r')

x = {}
for line in file1:
    if line in x:
        x[line]+=1
    else:
        x[line]=1


x1 = sorted(x.items(), key= lambda item:int(item[0].replace('\n','')));

'''
for a in x:
    if x[a]>1:
        print(a, x[a])
'''
for a in x1:
    if a[1]>0:
        print(a)
