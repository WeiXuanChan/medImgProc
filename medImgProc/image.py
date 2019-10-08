'''
File: image.py
Description: load all class for medImgProc
             Contains main image class
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         12JAN2018           - Created
  Author: w.x.chan@gmail.com         08OCT2018           - v1.4.2
                                                              -debug changeColorFormat

Requirements:
    numpy.py
    matplotlib.py
    imageio.py

Known Bug:
    HSV color format not supported
All rights reserved.
'''
_version='1.4.2'

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os
import matplotlib
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
#import threading
#import time
import imageio
try:
    import medpy.io
except:
    pass
import pickle
import multiprocessing
import interpolationScheme
import GUI
'''
VARIABLES
'''
DEFAULT_IMG_DIMENSION=('y','x','RGB') #default dimension in loading image
DEFAULT_VOL_DIMENSION=('z','y','x','RGB') #default dimension in loading volume
DEFAULT_VID_DIMENSION=('t','y','x','RGB') #default dimension in loading volume
DEFAULT_SEGMENTATION_DIMENSION=('t','z','y','x')
DEFAULT_INTERPOLATION_SCHEME=interpolationScheme.linearEquation
ZEROTHORDER_INTERPOLATION_SCHEME=interpolationScheme.nearestValue
HINT_DIMENSION_LIST='Invalid Dimension.\n Please input accurate dimension list with:\n dimension=[str,str,...]'
COLOR_DIMENSION=('RGB','HSV')
'''
Internal use functions
'''
def adjustPhyLengthToFirst(ImageA,ImageB):
    '''
    Adjust physical length (such as fps to meter per pixel etc) to first image
    !!!Warning: editing input class!!!
    '''
    ImageB.dimlen=ImageA.dimlen[:]
    return;
def dataInColorFormat(imageClass,colorFormat):
    '''
    Return image with color base on colorFormat
    '''
    if colorFormat in imageClass.dim:
        print('Error: Image is already in '+colorFormat+' color format.')
    elif colorFormat=='RGB':
        dimAcsCount=list(range(1,len(imageClass.data.shape)+1))
        imageArray=np.array([imageClass.data[:],imageClass.data[:],imageClass.data[:]])
        return imageArray.transpose(*dimAcsCount,0)
    else:
        print('Error: color format is not supported.')
def dataInGreyscaleFormat(imageClass):
    '''
    Return greyscale image:
    RGB: average RGB values
    '''
    if 'RGB' in imageClass.dim:
        RGBFirstIndex,newDim=arrangeDim(imageClass.dim,['RGB'],True)
        imageArrayWithColorFirst=imageClass.data.transpose(RGBFirstIndex)
        return np.mean(imageArrayWithColorFirst,axis=0)
    else:
        print('Error: Image is already in greyscale format or format not supported.')
def datatypeMinMax(dtype):
    maxI=float('inf')
    minI=float('-inf')
    if dtype==np.uint8:
        maxI=2**8-1
        minI=0
    elif dtype==np.uint16:
        maxI=2**16-1
        minI=0
    elif dtype==np.uint32:
        maxI=2**32-1
        minI=0
    elif dtype==np.int8:
        maxI=2**7-1
        minI=-(2**7)
    elif dtype==np.int16:
        maxI=2**15-1
        minI=-(2**15)
    elif dtype==np.int32:
        maxI=2**31-1
        minI=-(2**31)
    return (minI,maxI)
def normalizeArrayVal(oldArray,minVal,maxVal):
    arrayMax=oldArray.max()
    arrayMin=oldArray.min()
    return (oldArray-arrayMin)/(arrayMax-arrayMin)*(maxVal-minVal)+minVal
    
def arrangeDim(oldDim,newDim,arrangeFront):
    '''
    return the transpose index for oldDim -> newDim as well as the full newDim with len(oldDim)==len(newDim)
    arrangeFront (if true) stance excess dimension to the back of newDim
    newDim must be a subset of oldDim
    '''
    oldDimCopy=oldDim[:]
    oldIndex=list(range(len(oldDimCopy)))
    transposeIndex=[]
    if type(newDim)!=list:
        newDim=[newDim]
    for axis in newDim:
        index=oldDim.index(axis)
        transposeIndex.append(index)
        oldDimCopy.remove(axis)
        oldIndex.remove(index)
    if arrangeFront:
        currentDim=newDim[:]+oldDimCopy
        transposeIndex=transposeIndex+oldIndex
    else:
        currentDim=oldDimCopy+newDim[:]
        transposeIndex=oldIndex+transposeIndex
    return (transposeIndex,currentDim)
def stretchFirstDimension(oldArray,stretchSize,scheme):
    '''
    stretch the first dimension of array according to scheme
    '''
    oldShape=oldArray.shape
    newArray=np.zeros((stretchSize,*oldShape[1:]))
    for n in range(stretchSize):
        newArray[n]=scheme(float(n*(oldShape[0]-1))/(stretchSize-1),oldArray)
    return newArray
def stretchImg(imageClass,stretchDim,scheme):
    '''
    stretch image according to stretchDim = {axis:length,...}
    '''
    currentDim=imageClass.dim[:]
    newArray=np.copy(imageClass.data)
    for axis in stretchDim:
        if axis in currentDim:
            transposeIndex,currentDim=arrangeDim(currentDim,[axis],True)
            newArray=newArray.transpose(transposeIndex)
            newArray=stretchFirstDimension(newArray,stretchDim[axis],scheme)
    transposeIndex,currentDim=arrangeDim(currentDim,imageClass.dim[:],True)
    newArray=newArray.transpose(transposeIndex)
    return newArray
def compareDimSize(ImageA,ImageB):
    if len(ImageA.dim)>=len(ImageB.dim):
        return (ImageA,ImageB)
    else:
        return (ImageB,ImageA)
def numpyArithwrtImage(func,ImageA,ImageB):
    largerDim_image,smallerDim_image=compareDimSize(ImageA,ImageB)
    transposeIndex,currentDim=arrangeDim(largerDim_image.dim,smallerDim_image.dim,False)
    largerDim_image.data=largerDim_image.data.transpose(transposeIndex)
    if len(ImageA.dim)>=len(ImageB.dim):
        largerDim_image.data=func(largerDim_image.data,smallerDim_image.data)
    else:
        largerDim_image.data=func(smallerDim_image.data,largerDim_image.data)
    transposeIndex,currentDim=arrangeDim(currentDim,largerDim_image.dim,True)
    largerDim_image.data=largerDim_image.data.transpose(transposeIndex)
    return largerDim_image
def applyFuncRecursive(func,funcArgs,axes,currentDim,currentData,dimSlice):
    if currentDim!=axes:
        passArray=currentData[dimSlice[currentDim[0]]]
        if isinstance(dimSlice[currentDim[0]],int):
            passArray=np.array([passArray])
        for n in range(len(passArray)):
            passArray[n]=applyFuncRecursive(func,funcArgs,axes,currentDim[1:],passArray[n],dimSlice)
        if isinstance(dimSlice[currentDim[0]],int):
            passArray=passArray[0]
        currentData[dimSlice[currentDim[0]]]=passArray
    else:
        passSlice=[]
        for dimension in currentDim:
            passSlice.append(dimSlice[dimension])
        passSlice=tuple(passSlice)
        passArray=currentData[passSlice]
        passArray=func(passArray,*funcArgs)
        currentData[passSlice]=passArray
    return currentData
def applyFunc(imageClass,func,axes,dimSlice,funcArgs):#use slice(a,b) for otherDimLoc 
        newImage=imageClass.clone()
        for dimension in newImage.dim:#apply function to all 
            if dimension not in dimSlice:
                dimSlice[dimension]=-1
        for dimension in dimSlice: # -1 imply slice all
            if dimSlice[dimension]==-1:
                dimIndex=newImage.dim.index(dimension)
                dimSlice[dimension]=slice(newImage.data.shape[dimIndex])
        runData=np.copy(newImage.data)
        currentDim=newImage.dim[:]
        transposeIndex,currentDim=arrangeDim(currentDim,axes,False)
        runData=runData.transpose(transposeIndex)  
        newData=applyFuncRecursive(func,funcArgs,axes,currentDim[:],runData,dimSlice)
        transposeIndex,currentDim=arrangeDim(currentDim,newImage.dim,True)
        newData=newData.transpose(transposeIndex)
        return newData
def recursive2DWrite(imageArray,currentDim,axes,filePath,imageFormat,dimRange,fps=3,color=0):
    if currentDim[1:]!=axes:
        for n in dimRange[currentDim[0]]:
            newPath=os.path.normpath(filePath+'/'+currentDim[0]+str(n))
            os.makedirs(newPath, exist_ok=True)
            recursive2DWrite(imageArray[n],currentDim[1:],axes,newPath,imageFormat,dimRange,fps=fps,color=color)
    else:
        for n in dimRange[currentDim[0]]:
            if len(currentDim)==(4+color):
                imageio.mimwrite(os.path.normpath(filePath+'/'+currentDim[0]+str(n)+'.'+imageFormat),imageArray[n],format=imageFormat,fps=fps)
            elif len(currentDim)<(4+color):
                imageio.imwrite(os.path.normpath(filePath+'/'+currentDim[0]+str(n)+'.'+imageFormat),imageArray[n])

def linearSampling(image,pixelfloat,fill=0):
    return RegularGridInterpolator(tuple(map(range,image.data.shape)),image.data,fill_value=fill,bounds_error=False)(pixelfloat)

class gaussianSampling:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,imgDim,imgDimlen,amplitude=5.01,variance=None):
        '''
        Initialize all data.
        Note: If adding variables, add to clone() function too 
        '''
        self.amplitude=amplitude
        self.imgDim=imgDim
        if type(imgDimlen)==dict:
            self.imgDimlen=[]
            for dim in imgDim:
                self.imgDimlen.append(imgDimlen[dim])
            self.imgDimlen=np.array(self.imgDimlen)
        else:
            self.imgDimlen=np.array(imgDimlen)
        if type(variance)==type(None):
            self.variance=np.ones(3)*max(self.imgDimlen)**2.
        elif type(variance)==dict:
            self.variance=[]
            for dim in imgDim:
                self.variance.append(variance[dim])
            self.variance=np.array(self.variance)
        elif type(variance) in [float,int]:
            self.variance=np.zeros(3)
            print(imgDim,variance,self.imgDimlen)
            for dimN in range(len(imgDim)):
                self.variance[dimN]=variance*self.imgDimlen[dimN]**2.
        else:
            self.variance=variance*self.imgDimlen**2.
        cropDistance=np.sqrt(2.*self.variance*np.log(amplitude/0.5))
        matlen=[]
        for ndim in range(len(imgDim)):
            matlen.append(int(np.ceil(cropDistance[ndim]/self.imgDimlen[ndim])))
        self.baseCoordMat = np.mgrid[(-matlen[0]*self.imgDimlen[0]):((matlen[0]+1.1)*self.imgDimlen[0]):self.imgDimlen[0], (-matlen[1]*self.imgDimlen[1]):((matlen[1]+1.1)*self.imgDimlen[1]):self.imgDimlen[1],(-matlen[2]*self.imgDimlen[2]):((matlen[2]+1.1)*self.imgDimlen[2]):self.imgDimlen[2]].reshape(3,*((np.array(matlen)+1)*2)).transpose(1,2,3,0)
        disSqList=np.sum((self.baseCoordMat)**2./2./self.variance,axis=3).reshape(-1)
        filterInd=np.nonzero(disSqList<np.log(2.*self.amplitude))
        disSq=np.sum((self.baseCoordMat.reshape((-1,3))[filterInd])**2./2./self.variance.max(),axis=1)
        countList=np.around(self.amplitude*np.exp(-disSq)).astype(int)
        print('baseCoordMat shape',self.baseCoordMat.shape)
        print('variance=',self.variance)
        print('spread=',len(countList),', self weighted=',float(max(countList))/np.sum(countList))
        print(countList)
    def __call__(self,image,pixelfloat,fill=0):
        coordAdjust=(np.array(pixelfloat[:3])%1)*self.imgDimlen
        disSqList=np.sum((self.baseCoordMat-coordAdjust)**2./2./self.variance,axis=3).reshape(-1)
        filterInd=np.nonzero(disSqList<np.log(2.*self.amplitude))
        coordfloor=np.floor(np.array(pixelfloat[:3]))*self.imgDimlen
        coordList=(self.baseCoordMat+coordfloor).reshape((-1,3))
        coordMat=np.hstack((coordList,np.ones((len(coordList),1))*pixelfloat[3]))
        countList=np.around(self.amplitude*np.exp(-disSqList[filterInd])).astype(int)
        valueList=image.getData(coordMat[filterInd],fill=fill)
        value=[]
        for n in range(len(valueList)):
            value+=[valueList[n]]*countList[n]
        #print(len(countList),countList)
        return value
        
        
'''
Image Class
'''
class image:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,imageFile=None,dimension=None,fileFormat='',dimlen={},crop=None,module=''):
        '''
        Initialize all data.
        Note: If adding variables, add to clone() function too 
        '''
        self.data=None
        self.dim=None
        self.dtype=None
        self.dimlen=None
        if imageFile is not None:
            if type(imageFile)==str:
                self.imread(imageFile,dimension=dimension,fileFormat=fileFormat,dimlen=dimlen,module=module)
            elif type(imageFile)==np.ndarray:
                self.data=imageFile
            if crop:
                toSlice=[]
                for slicing in crop:
                    if type(slicing)==slice:
                        toSlice.append(slicing)
                    elif type(slicing)==list:
                        toSlice.append(slice(*slicing))
                    elif type(slicing)==int:
                        toSlice.append(slice(slicing))
                    else:
                        print('crop error, no slicing with input:',slicing) 
                        toSlice.append(slice(None))
                self.data=self.data[tuple(toSlice)]
    def clone(self):
        new_image=image()
        new_image.data=np.copy(self.data)
        new_image.dim=self.dim[:]
        new_image.dimlen=self.dimlen.copy()
        new_image.dtype=self.dtype
        return new_image
    '''
    Arithmetics C = A +-*/ B :
    C will try to keep the shape of A, dimension will depend on the larger dimension.
    A.dim must be a subset of B.dim or B.dim is a subset of A.dim
    Warning!! dimension length is ignored.
    '''
    def __add__(self, other,reportErrorOnFail=False):
        new_image=self.clone()
        if isinstance(other, (int, float)):
            new_image.data=new_image.data+other
            return new_image
        if new_image.dim==other.dim:
            if new_image.data.shape==other.data.shape:
                new_image.data=self.data+other.data
                return new_image
        new_other=other.stretch(dict(zip(new_image.dim,new_image.data.shape)))
        adjustPhyLengthToFirst(new_image,new_other)
        result=numpyArithwrtImage(np.add,new_image,new_other)
        return result
        
        
    def __radd__(self, other):
        if isinstance(other, (int, float)):
            new_image=self.clone()
            new_image.data=other+self.data
            return new_image
        else:
            new_image=other.__add__(new_image)
            return new_image
    def __sub__(self, other,reportErrorOnFail=False):
        new_image=self.clone()
        if isinstance(other, (int, float)):
            new_image.data=self.data-other
            return new_image
        if self.dim==other.dim:
            if self.data.shape==other.data.shape:
                new_image.data=self.data-other.data
                return new_image
        new_other=other.stretch(dict(zip(new_image.dim,new_image.data.shape)))
        adjustPhyLengthToFirst(new_image,new_other)
        result=numpyArithwrtImage(np.sub,new_image,new_other)
        return result
    def __rsub__(self, other):
        new_image=self.clone()
        if isinstance(other, (int, float)):
            new_image.data=other-self.data
            return new_image
        if self.dim==other.dim:
            if self.data.shape==other.data.shape:
                new_image.data=other.data-self.data
                return new_image
        new_other=other.clone()
        new_image=new_image.stretch(dict(zip(new_other.dim,new_other.data.shape)))
        adjustPhyLengthToFirst(new_other,new_image)
        result=numpyArithwrtImage(np.sub,new_other,new_image)        
        return result
    def __mul__(self, other):
        new_image=self.clone()
        if isinstance(other, (int, float)):
            new_image.data=self.data*other
            return new_image
        elif self.dim==other.dim:
            if self.data.shape==other.data.shape:
                new_image.data=self.data*other.data
                return new_image
        new_other=other.stretch(dict(zip(new_image.dim,new_image.data.shape)))
        adjustPhyLengthToFirst(new_image,new_other)
        result=numpyArithwrtImage(np.multiply,new_image,new_other)
        return result
    def __rmul__(self, other):
        new_image=self.clone()
        if isinstance(other, (int, float)):
            new_image.data=other*self.data
            return new_image
        elif self.dim==other.dim:
            if self.data.shape==other.data.shape:
                new_image.data=other.data*self.data
                return new_image
        new_other=other.clone()
        new_image=new_image.stretch(dict(zip(new_other.dim,new_other.data.shape)))
        adjustPhyLengthToFirst(new_other,new_image)
        result=numpyArithwrtImage(np.multiply,new_other,new_image)        
        return result
    def __truediv__(self, other):
        new_image=self.clone()
        if isinstance(other, (int, float)):
            new_image.data=self.data/other
            return new_image
        elif self.dim==other.dim:
            if self.data.shape==other.data.shape:
                new_image.data=self.data/other.data
                return new_image
        new_other=other.stretch(dict(zip(new_image.dim,new_image.data.shape)))
        adjustPhyLengthToFirst(new_image,new_other)
        result=numpyArithwrtImage(np.divide,new_image,new_other)
        return result
    def __rtruediv__(self, other):
        new_image=self.clone()
        if isinstance(other, (int, float)):
            new_image.data=other/self.data
            return new_image
        elif self.dim==other.dim:
            if self.data.shape==other.data.shape:
                new_image.data=other.data/self.data
                return new_image
        new_other=other.clone()
        new_image=new_image.stretch(dict(zip(new_other.dim,new_other.data.shape)))
        adjustPhyLengthToFirst(new_other,new_image)
        result=numpyArithwrtImage(np.divide,new_other,new_image)        
        return result
    def __pow__(self, val):
        new_image=self.clone()
        new_image.data=self.data**val
        return new_image
    def __rpow__(self, val):
        new_image=self.clone()
        new_image.data=val**self.data
        return new_image
    def __neg__(self):
        new_image=self.clone()
        new_image.data=-new_image.data
        return new_image
    def getData(self,coordList,fill=0,sampleFunction=None,CPU=1,getResidual=False):
        '''
        intensityValueList=sampleFunction(self,coordinate) 
        '''
        
        singleInput=False
        if not(type(coordList[0]) in [np.ndarray,list,dict]):
            coordList=[coordList]
            singleInput=True
        if CPU>1:
            fxstarmapInput=np.empty( (len(coordList),3), dtype=object)
            fxstarmapInput[:,0]=list(coordList)
            fxstarmapInput[:,1]=fill
            fxstarmapInput[:,2]=sampleFunction
            pool = multiprocessing.Pool(CPU)
            dataList=pool.starmap(self.getData,fxstarmapInput)
            pool.close()
            pool.join()
        else:
            dataList=[]
            for n in range(len(coordList)):
                coord=[]
                if type(coordList[n])==dict:
                    for dim in self.dim:
                        coord.append(coordList[n][dim]/self.dimlen[dim])
                        #coord.append(int(np.around(coordList[n][dim]/self.dimlen[dim])))
                else:
                    for dimN in range(len(coordList[n])):
                        coord.append(coordList[n][dimN]/self.dimlen[self.dim[dimN]])
                        #coord.append(int(np.around(coordList[n][dimN]/self.dimlen[self.dim[dimN]])))
                value=fill
                coordInd=[]
                if type(sampleFunction)==type(None):
                    for m in range(len(coord)):
                        tempcoord=int(np.around(coord[m]))
                        if tempcoord<0:
                            break
                        if tempcoord>=self.data.shape[m]:
                            break
                        coordInd.append(tempcoord)
                    else:
                        value=self.data[tuple(coordInd)]
                else:
                    value=sampleFunction(self,coord,fill)
                if getResidual and type(sampleFunction)==type(None):
                    if len(coord)==len(coordInd):
                        dataList.append([value,np.array(coord)-np.array(coordInd)])
                    else:
                        dataList.append([value,np.zeros(len(coord))])
                else:
                    dataList.append(value)
        if singleInput:
            dataList=dataList[0]
        
        return dataList
            
    '''
    Saving data (readable)
    '''
    def imwrite2D(self,filePath,axes=['y','x'],imageFormat='png',dimRange={},fps=3,color=0):
        os.makedirs(filePath, exist_ok=True)
        if axes[-1] in ['RGB','RGBA']:
            color=1
        saveData=np.copy(self.data)
        currentDim=self.dim[:]
        for dimension in currentDim:#apply function to all 
            if dimension not in dimRange:
                dimRange[dimension]=-1
        for dimension in dimRange: # -1 imply slice all
            if dimRange[dimension]==-1:
                dimIndex=currentDim.index(dimension)
                dimRange[dimension]=range(saveData.shape[dimIndex])
        transposeIndex,currentDim=arrangeDim(currentDim,axes,False)
        saveData=saveData.transpose(transposeIndex)
        if len(currentDim)==len(axes):
            if len(currentDim)==(3+color):
                imageio.mimwrite(os.path.normpath(filePath+'/0.'+imageFormat),saveData,format=imageFormat,fps=fps)
            elif len(currentDim)==(2+color):
                imageio.imwrite(os.path.normpath(filePath+'/0.'+imageFormat),saveData)
            print(currentDim)
        else:
            recursive2DWrite(saveData,currentDim,axes,filePath,imageFormat,dimRange,fps=fps,color=color)
        dimlen_np=[]
        for ti in transposeIndex:
            dimlen_np.append(self.dimlen[self.dim[ti]])
        dimlen_np=np.array(dimlen_np)
        np.savetxt(os.path.normpath(filePath+'/dimensionLength.txt'),dimlen_np)
        print('Image written to:',filePath)
        #self.save(os.path.normpath(filePath+'/image.mip'))
    def mimwrite2D(self,filePath,axes=['t','y','x'],vidFormat='gif',dimRange={},fps=3):
        self.imwrite2D(filePath,axes=axes,imageFormat=vidFormat,dimRange=dimRange,fps=fps)
    '''
    saving and loading object
    '''
    def save(self,file):
        with open(file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    '''
    Functions which changes class data
    '''
    def setDefaultDimlen(self):
        if self.dim is not None:
            for dimension in self.dim:
                if not(dimension in self.dimlen) and not(dimension in COLOR_DIMENSION):
                    self.dimlen[dimension]=1.
        else:
            self.dimlen={}
    def addDim(self,newDim,newDimLen=1.):
        self.dim.insert(0,newDim)
        self.data=np.array([self.data])
        self.dimlen[newDim]=newDimLen
    def removeDim(self,rDim,refDim=0):
        ind=self.dim.index(rDim)
        if ind>0:
            sliceind=[]
            for n in range(ind):
                sliceind.append(slice(None))
            sliceind.append(refDim)
        else:
            sliceind=[refDim]
        self.data=self.data[tuple(sliceind)].copy()
        del self.dimlen[rDim]
        self.dim.pop(ind)
    def rearrangeDim(self,newDim,arrangeFront=True):
        transposeIndex,currentDim=arrangeDim(self.dim[:],newDim,arrangeFront)
        self.data=self.data.transpose(transposeIndex)
        self.dim=currentDim[:]
        return transposeIndex
    def insertnewImage(self,newImage,stackDim,index=-1):
        currentDim=self.dim[:]
        newArray=np.copy(self.data)
        transposeIndex,currentDim=arrangeDim(currentDim,[stackDim],True)
        newArray=newArray.transpose(transposeIndex)
        
        transposeIndex,newImageArrangedDim=arrangeDim(newImage.dim,currentDim[1:],True)
        if newImageArrangedDim==currentDim[1:]:
            newImageArray=newImage.data.transpose(transposeIndex)
            if newImageArray.shape!=newArray.shape[1:]:#expand shape
                newshape=[]
                newImageArraySlice=[]
                newArraySlice=[slice(None)]
                for n in range(len(newImageArray.shape)):
                    newshape.append(max(newImageArray.shape[n],newArray.shape[n+1]))
                    newArraySlice.append(slice(newArray.shape[n+1]))
                    newImageArraySlice.append(slice(newImageArray.shape[n]))
                tempNewArray=np.zeros((newArray.shape[0],*newshape),dtype=newArray.dtype)
                tempNewArray[tuple(newArraySlice)]=newArray
                newArray=tempNewArray.copy()
                tempNewImageArray=np.zeros(newshape,dtype=newImageArray.dtype)
                tempNewImageArray[tuple(newImageArraySlice)]=newImageArray
                newImageArray=tempNewImageArray.copy()
                
            if index<0:
                newArray=np.array([*newArray[:(len(newArray)+1+index)],newImageArray,*newArray[(len(newArray)+1+index):]])
            else:
                newArray=np.array([*newArray[:index],newImageArray,*newArray[index:]])
            transposeIndex,currentDim=arrangeDim(currentDim,self.dim,True)
            newArray=newArray.transpose(transposeIndex)
            self.data=newArray
        else:
            print('Error!!! Stacking Image '+newImage+' of different dimension')
    def insertnewImageList(self,newImageList,stackDim):
        for newImage in newImageList:
            self.insertnewImage(newImage,stackDim)
    def applyFunction(self,func,axes=['y','x'],dimSlice={},funcArgs=()):#use slice(a,b) for otherDimLoc 
        newArray=applyFunc(self,func,axes,dimSlice,funcArgs)
        self.data=newArray
        return self
    def stretch(self,stretchDim,scheme=DEFAULT_INTERPOLATION_SCHEME,stretchData=True):
        for dimension in stretchDim:
            stretchRatio=(self.data.shape[self.dim.index(dimension)]-1)/(stretchDim[dimension]-1)
            self.dimlen[dimension]=self.dimlen[dimension]*stretchRatio
        if stretchData:
            newArray=stretchImg(self,stretchDim,scheme)
            minI,maxI=datatypeMinMax(self.data.dtype)
            minVal=newArray.min()
            maxVal=newArray.max()
            if newArray.min()<minI:
                minVal=minI
            if newArray.max()>maxI:
                maxVal=maxI
            newArray=normalizeArrayVal(newArray,minVal,maxVal)
            
            self.data=newArray.astype(self.data.dtype)
    def normalize(self,minVal,maxVal):
        newArray=normalizeArrayVal(self.data,minVal,maxVal)
        self.data=newArray.astype(self.data.dtype)
    def changeColorFormat(self,colorFormat='RGB'):
        new_data=dataInColorFormat(self,colorFormat)
        if new_data is not None:
            self.data=new_data
            self.dim.append(colorFormat)
    def changeGreyscaleFormat(self,colorFormat='RGB'):
        new_data=dataInGreyscaleFormat(self)
        if new_data is not None:
            self.data=new_data
            self.dim.remove(colorFormat)
    def invert(self):
        minI,maxI=datatypeMinMax(self.dtype)
        self.data=maxI+minI-self.data
    def empty(self):
        minI,maxI=datatypeMinMax(self.dtype)
        newArray=np.ones(self.data.shape)*minI
        newArray.astype(self.dtype)
        self.data=newArray
    def intensityShift(self,shift,minI=None,maxI=None):
        if minI is None:
            minI=self.data.min()
        if maxI is None:
            maxI=self.data.max()
        self.data=np.clip(self.data, minI-shift, maxI-shift)
        self.data=self.data+shift
    def imread(self,imageFile,dimension=None,fileFormat='',dimlen={},module=''):
        '''
        Identify input file as Image
        '''
        if module=='medpy':
            try:
                img, img_header = medpy.io.load(os.path.normpath(imageFile))
            except:
                pass
            else:
                dim=len(img.shape)
                self.data=np.array(img)
                if dimension is None:
                    dimension=list(DEFAULT_IMG_DIMENSION)
                    if img.shape[-1]>4 and len(dimension)==dim:
                        dimension=list(DEFAULT_VOL_DIMENSION)
                self.dim=dimension
                self.dtype=img.dtype
                if dimlen=={}:
                    self.dimlen={}
                    for dimN in range(len(img_header.get_voxel_spacing())):
                        self.dimlen[dimension[dimN]]=img_header.get_voxel_spacing()[dimN]
                else:
                    self.dimlen=dimlen
                return
        try:
            img=imageio.imread(os.path.normpath(imageFile))
        except:
            pass
        else:
            dim=len(img.shape)
            if dimension is None:
                dimension=list(DEFAULT_IMG_DIMENSION)
                if img.shape[-1]>4 and len(dimension)==dim:
                    dimension=list(DEFAULT_VOL_DIMENSION)
            if len(dimension)<dim:
                print('Error loading image file.'+HINT_DIMENSION_LIST)
                return
            else:
                self.data=np.array(img)
                self.dim=dimension[:dim]
                self.dimlen=dimlen
                self.dtype=img.dtype
                self.setDefaultDimlen()
                return
        '''
        Identify input file as Volume
        '''   
        try:
            if fileFormat =='':
                img=imageio.volread(os.path.normpath(imageFile))
            else:
                img=imageio.volread(os.path.normpath(imageFile),fileFormat)
        except:
            pass
        else:
            dim=len(img.shape)
            if dimension is None:
                dimension=list(DEFAULT_VOL_DIMENSION)
            if len(dimension)<dim:
                print('Error loading volume file.'+HINT_DIMENSION_LIST)
                return
            else:
                self.data=np.array(img)
                self.dim=dimension[:dim]
                self.dimlen=dimlen
                self.dtype=img.dtype
                self.setDefaultDimlen()
                return
        '''
        Identify input file as Video
        ''' 
        try:
            img=imageio.get_reader(os.path.normpath(imageFile))
        except:
            pass
        else:
            frames=img.get_meta_data()['nframes']
            if dimension is None:
                dimension=list(DEFAULT_VID_DIMENSION)
            for vid_frame in img: #access frame dimension and dtype
                frameShape=vid_frame.shape
                dim=len(frameShape)+1
                self.dtype=vid_frame.dtype
                break
            if len(dimension)<dim:
                self.dtype=None
                print('Error loading video file.'+HINT_DIMENSION_LIST)
                return
            else:
                self.data=np.zeros((frames,*frameShape),dtype=self.dtype)
                self.dim=dimension[:dim]
                self.dimlen=dimlen
                self.dimlen[dimension[0]]=img.get_meta_data()['fps']
                self.setDefaultDimlen()
                countFrame=0
                for vid_frame in img:
                    self.data[countFrame]=np.array(vid_frame)
                    countFrame+=1
                return
        raise Exception('Error reading file')
        return
    
    '''

    graphical user interface
    '''
    def show(self,tag='2D'):
        if tag=='2D':
            GUI.image2DGUI(self,disable=['click'])
    def readPoints(self,disable=['swap'],addInstruct=''):
        result=GUI.image2DGUI(self,disable=disable,addInstruct=addInstruct)
        transposeIndex,currentDim=arrangeDim(result.image.dim,self.dim,True)
        points=result.points[:,transposeIndex]
        return points
        

