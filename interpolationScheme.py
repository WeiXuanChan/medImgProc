'''
File: interpolationScheme.py
Description: library of all interpolation scheme
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         12JAN2018           - Created

Requirements:
    numpy.py

All rights reserved.
'''

import numpy as np

def CRCHsplineEquation(floatIndex,refPoints):
    n=int(floatIndex)
    if n==0:
        s0=0.
        s3=2.
        p0=refPoints[n].astype(float)
        p3=refPoints[n+2].astype(float)
    elif n==(len(refPoints)-2):
        s0=-1.
        s3=1
        p0=refPoints[n-1].astype(float)
        p3=refPoints[n+1].astype(float)
    elif n==(len(refPoints)-1):
        return np.copy(refPoints[n])
    else:
        s0=-1.
        s3=2.
        p0=refPoints[n-1].astype(float)
        p3=refPoints[n+2].astype(float)
    
    s1=0.
    s2=1.
    p1= refPoints[n].astype(float)
    p2= refPoints[n+1].astype(float)

    m1=(p2-p0)/2.
    m2=(p3-p1)/2.
    t=floatIndex-n
    p=(2.*t**3.-3.*t**2.+1)*p1+(t**3.-2.*t**2.+t)*m1+(-2.*t**3+3.*t**2.)*p2+(t**3.-t**2.)*m2

    return np.copy(p)

def linearEquation(floatIndex,refPoints):
    n=int(floatIndex)
    if n==(len(refPoints)-1):
        return np.copy(refPoints[n])
    
    p1= refPoints[n].astype(float)
    p2= refPoints[n+1].astype(float)
    t=floatIndex-n
    p=t*(p2-p1)+p1
    return np.copy(p)
def nearestValue(floatIndex,refPoints):
    n=int(np.around(floatIndex))
    if n==(len(refPoints)-1):
        return np.copy(refPoints[n])
    
    p= refPoints[n].astype(float)
    return np.copy(p)
