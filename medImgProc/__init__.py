
'''
################################################
MIT License

Copyright (c) 2019 W. X. Chan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
################################################
File: __init__.py
Description: load all class for medImgProc
             Contains externally usable class
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         12JAN2019           - Created
  Author: w.x.chan@gmail.com         12JAN2019           - v1.3.2
                                                            -processFunc v1.3.0
                                                            -Image v1.0.0
                                                            -GUI v1.0.0
  Author: w.x.chan@gmail.com         12JAN2019           - v1.4.2
                                                            -processFunc v1.3.0
                                                            -Image v1.4.2
                                                            -GUI v1.4.0
  Author: w.x.chan@gmail.com         08OCT2019           - v1.5.2
                                                            -processFunc v1.3.0
                                                            -Image v1.4.2
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         08OCT2019           - v1.5.4
                                                            -processFunc v1.3.0
                                                            -Image v1.5.4
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         15OCT2019           - v1.5.5
                                                            -processFunc v1.3.0
                                                            -Image v1.5.5
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         15OCT2019           - v1.6.1
                                                                -added combine grescale image to color
                                                            -processFunc v1.3.0
                                                            -Image v1.5.5
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         29OCT2019           - v1.6.3
                                                                -added point based combine to function combine
                                                            -processFunc v1.3.0
                                                            -Image v1.6.2
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         29OCT2019           - v1.6.4
                                                            -processFunc v1.3.0
                                                            -Image v1.6.4
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         29OCT2019           - v1.6.5
                                                            -processFunc v1.3.0
                                                            -Image v1.6.5
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         30OCT2019           - v1.6.6
                                                            -processFunc v1.6.6
                                                            -Image v1.6.5
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         30OCT2019           - v1.7.0
                                                            -processFunc v1.7.0
                                                            -Image v1.6.5
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         31OCT2019           - v1.7.3
                                                            -processFunc v1.7.3
                                                            -Image v1.6.5
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         08NOV2019           - v1.7.6
                                                            -processFunc v1.7.3
                                                            -Image v1.7.6
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         08NOV2019           - v1.8.0
                                                            -processFunc v1.7.3
                                                            -Image v1.8.0
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         13NOV2019           - v1.8.5
                                                            -processFunc v1.8.5
                                                            -Image v1.8.0
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         13NOV2019           - v1.9.1
                                                            -processFunc v1.9.1
                                                            -Image v1.8.0
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         13NOV2019           - v2.0.0
                                                            -processFunc v2.0.0
                                                            -Image v1.8.0
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         18NOV2019           - v2.1.7
                                                            -processFunc v2.1.5
                                                            -Image v1.8.0
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         02JAN2020           - v2.2.8
                                                            -processFunc v2.2.8
                                                            -Image v1.8.0
                                                            -GUI v1.5.2
  Author: w.x.chan@gmail.com         10JAN2020           - v2.3.11
                                                            -processFunc v2.2.8
                                                            -Image v1.8.0
                                                            -GUI v2.3.10
  Author: w.x.chan@gmail.com         15JAN2020           - v2.3.13
                                                            -allow loadStack to add new dimension and stack at different axis
                                                            -processFunc v2.2.8
                                                            -Image v2.3.13
                                                            -GUI v2.3.10
  Author: w.x.chan@gmail.com         21JAN2020           - v2.3.14
                                                            -processFunc v2.3.14
                                                            -Image v2.3.13
                                                            -GUI v2.3.10
  Author: w.x.chan@gmail.com         21JAN2020           - v2.4.0
                                                            -processFunc v2.4.0
                                                            -Image v2.3.13
                                                            -GUI v2.3.10
                                                            -pointSpeckleProc v2.4.0
                                                            -Snake v2.4.0
  Author: w.x.chan@gmail.com         21JAN2020           - v2.4.3
                                                            -processFunc v2.4.0
                                                            -Image v2.4.1
                                                            -GUI v2.3.10
                                                            -pointSpeckleProc v2.4.0
  Author: w.x.chan@gmail.com         26FEB2020           - v2.4.6
                                                            -processFunc v2.4.6
                                                            -Image v2.4.1
                                                            -GUI v2.3.10
                                                            -pointSpeckleProc v2.4.0
  Author: w.x.chan@gmail.com         28FEB2020           - v2.5.0
                                                            -processFunc v2.5.0
                                                            -Image v2.4.1
                                                            -GUI v2.3.10
                                                            -pointSpeckleProc v2.4.0
  Author: w.x.chan@gmail.com         05Mar2020           - v2.5.2
                                                            -processFunc v2.5.1
                                                            -Image v2.4.1
                                                            -GUI v2.3.10
                                                            -pointSpeckleProc v2.4.0
  Author: w.x.chan@gmail.com         17Mar2020           - v2.5.7
                                                            -processFunc v2.5.7
                                                            -Image v2.4.1
                                                            -GUI v2.3.10
                                                            -pointSpeckleProc v2.4.0
  Author: w.x.chan@gmail.com         24Mar2020           - v2.6.13 - in combine, transfer boolean image to contour 
                                                            -processFunc v2.5.7
                                                            -Image v2.6.10
                                                            -GUI v2.6.13
                                                            -pointSpeckleProc v2.4.0
  Author: w.x.chan@gmail.com         31Mar2020           - v2.6.18
                                                            -processFunc v2.6.18
                                                            -Image v2.6.15
                                                            -GUI v2.6.13
                                                            -pointSpeckleProc v2.4.0
  Author: w.x.chan@gmail.com         29Apr2020           - v2.6.19
                                                            -processFunc v2.6.18
                                                            -Image v2.6.15
                                                            -GUI v2.6.19
                                                            -pointSpeckleProc v2.4.0
  Author: w.x.chan@gmail.com         21Jul2020           - v2.6.24
                                                            -processFunc v2.6.24
                                                            -Image v2.6.15
                                                            -GUI v2.6.19
                                                            -pointSpeckleProc v2.4.0
  Author: w.x.chan@gmail.com         25Aug2020           - v2.6.25
                                                            -processFunc v2.6.24
                                                            -Image v2.6.25
                                                            -GUI v2.6.19
                                                            -pointSpeckleProc v2.4.0
  Author: w.x.chan@gmail.com         09Oct2020           - v2.6.28
                                                            -processFunc v2.6.28
                                                            -Image v2.6.25
                                                            -GUI v2.6.19
                                                            -pointSpeckleProc v2.4.0
  Author: w.x.chan@gmail.com         09Oct2020           - v2.6.30
                                                            -processFunc v2.6.30
                                                            -Image v2.6.25
                                                            -GUI v2.6.19
                                                            -pointSpeckleProc v2.4.0
  Author: w.x.chan@gmail.com         10Oct2020           - v2.6.31
                                                            -processFunc v2.6.31
                                                            -Image v2.6.25
                                                            -GUI v2.6.19
                                                            -pointSpeckleProc v2.4.0


Requirements:
    numpy.py
    matplotlib.py
    imageio.py

Known Bug:
    HSV color format not supported
All rights reserved.
'''
import logging
_version='2.6.31'
logger = logging.getLogger('medImgProc v'+_version)
logger.info('medImgProc version '+_version)


import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import matplotlib
if os.environ.get('DISPLAY','')=='' and os.name!='nt':
    logger.warning('no display found. Using non-interactive Agg backend')
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
    try:
        with open(filePath, 'rb') as input:
            outObj = pickle.load(input)
    except:
        outObj=imread(filePath)
    return outObj
def loadStack(imageFileFormat,dimension=None,n=0,maxskip=0):
    getFunc=None
    if isinstance(n,(list,tuple)):
        if len(n)==0:
            n=0
            n_new=0
        elif len(n)==1:
            n=n[0]
            n_new=n
        else:
            n_new=list(n)
            n=n_new.pop(0)
    else:
        n_new=n
    if isinstance(dimension,(list,tuple)):
        dimension_new=[]
        if len(dimension)==0:
            dimension=None
        else:
            dimension_new=list(dimension)
            dimension=dimension_new.pop(0)
        if len(dimension_new)>0:
            getFunc=lambda x_imageFileFormat : loadStack(x_imageFileFormat,dimension=dimension_new,n=n_new,maxskip=maxskip)
            newImage=getFunc(imageFileFormat.format(n))
    if type(getFunc)==type(None):
        try:
            newImage=load(imageFileFormat.format(n))
            getFunc=load
        except:
            newImage=imread(imageFileFormat.format(n))
            getFunc=imread
    if type(dimension)==type(None):
        stackaxis=0
    elif isinstance(dimension,int):
        stackaxis=dimension
    elif dimension in newImage.dim:
        stackaxis=newImage.dim.index(dimension)
    else:
        newImage.addDim(dimension)
        stackaxis=0
    n+=1
    skip=0
    while True:
        try:
            nextImg=getFunc(imageFileFormat.format(n))
            if dimension not in nextImg.dim:
                nextImg.data=nextImg.data.reshape((1,*nextImg.data.shape))
            newImage.data=np.concatenate((newImage.data,nextImg.data.copy()),axis=stackaxis)
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

def combine(img1,img2,img3=None,point=False):
    #img1 is main image
    img=img1.clone()
    if img1.data.dtype==bool:
        img.changeBooleanToContour()
    img.changeColorFormat()
    if type(img3)!=type(None):
        if img2.data.dtype==bool:
            img2=img2.clone()
            img2.changeBooleanToContour()
        if img3.data.dtype==bool:
            img3=img3.clone()
            img3.changeBooleanToContour()
        img.data[...,1]=img2.data.copy()
        img.data[...,2]=img3.data.copy()
    elif point:
        img.data[...,0][img2.data>0]=img2.data[img2.data>0]
        img.data[...,1][img2.data>0]=255
        img.data[...,2][img2.data>0]=0
    else:
        if img2.data.dtype==bool:
            img2=img2.clone()
            img2.changeBooleanToContour()
        img.data[...,0]=np.maximum(img2.data,img1.data)
    return img
