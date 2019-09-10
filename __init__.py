'''
File: __init__.py
Description: load all class for medImgProc
             Contains externally usable class
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         12JAN2018           - Created

Requirements:
    numpy.py
    matplotlib.py
    imageio.py

Known Bug:
    HSV color format not supported
All rights reserved.
'''
print('medImgProc version 1.2.2')


import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import matplotlib
if os.environ.get('DISPLAY','')=='':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import pickle
try:
    import scipy.io as sio
except:
    pass

import image

'''
External Functions
'''
def imread(imageFile,dimension=None,fileFormat='',crop=None,module=''):
    newImage=image.image(imageFile,dimension=dimension,fileFormat=fileFormat,crop=crop,module=module)
    return newImage
def imwrite(imageClass,filePath,axes=['y','x'],imageFormat='png',dimRange={},fps=3,color=0):
    imageClass.imwrite2D(filePath,axes=axes,imageFormat=imageFormat,dimRange=dimRange,fps=fps,color=color)
def stackImage(imageList,newDim):
    newImage=imageList[0].clone()
    newImage.addDim(newDim)
    newImage.insertnewImageList(imageList[1:],newDim)
    return newImage
def apply(imageClass,func,axes=['y','x'],dimSlice={},funcArgs=()):#use slice(a,b) for otherDimLoc 
    newImage=image.applyFunc(imageClass,func,axes,dimSlice,funcArgs)
    return newImage
def arrange(imageClass,newDim,arrangeFront=True):
    newImage=imageClass.clone()
    transposeIndex,currentDim=image.arrangeDim(newImage.dim[:],newDim,arrangeFront)
    newImage.data=newImage.data.transpose(transposeIndex)
    newImage.dim=currentDim[:]
    return newImage
def stretch(imageClass,stretchDim,scheme=image.DEFAULT_INTERPOLATION_SCHEME):
    newImage=imageClass.clone()
    newImage.stretch(stretchDim,scheme=scheme)
    return newImage
def save(imageClass,filePath):
    imageClass.save(filePath)
def load(filePath):
    with open(filePath, 'rb') as input:
        outObj = pickle.load(input)
    return outObj
def loadStack(imageFileFormat,dimension=None,maxskip=0):
    newImage=load(imageFileFormat.format(0))
    n=1
    skip=0
    while True:
        try:
            newImage.data=np.concatenate((newImage.data,load(imageFileFormat.format(n)).data),axis=0)
            skip=0
        except:
            skip+=1
            if skip>maxskip:
                break
            print(n)
        n+=1
    return newImage
def loadmat(fileName,arrayName='',dim=[],dimlen={},dtype=None):
    newImage=image.image()
    try:
        matVariables=sio.loadmat(fileName)
        if not(arrayName):
            for key in matVariables:
                if type(matVariables[key])==np.ndarray:
                    newImage.data=matVariables[key]
                    break
            else:
                print('Error Loading matlab file.')
                return
    except NotImplementedError:
        import h5py
        matVariables = h5py.File(fileName)
        if not(arrayName):
            for key in matVariables:
                print(type(matVariables[key].value))
                if type(matVariables[key].value)==np.ndarray:
                    newImage.data=matVariables[key].value
                    break
            else:
                print('Error Loading matlab file.')
                return
    if len(dim)!=len(newImage.data.shape):
        newImage.dim=image.DEFAULT_SEGMENTATION_DIMENSION[-len(newImage.data.shape):]
    else:
        newImage.dim=dim[:]
    newImage.dimlen=dict(dimlen)
    if len(newImage.dim)!=len(newImage.dimlen):
        for dimension in newImage.dim:
            if dimension not in newImage.dimlen:
                newImage.dimlen[dimension]=1.
    if dtype is None:
        newImage.dtype=newImage.data.dtype
    else:
        newImage.data=newImage.data.astype(dtype)
        newImage.dtype=dtype
    return newImage
def show(imageClass):
    imageClass.show()
    
