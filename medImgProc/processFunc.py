'''
File: processFunc.py
Description: various function to process image
             
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         12JAN2018           - Created
    Author: w.x.chan@gmail.com         13SEP2018           - v1.3.0
                                                              -addded resolutionLevel for TmapRegister
                                                              -added twoD for TmapRegister_img2img

Requirements:
    numpy.py
    scipy.py
    cv2.py (opencv)

Known Bug:
    color format not supported
    last point of first axis ('t') not recorded in snapDraw_black
All rights reserved.
'''
_version='1.3.0'

import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from scipy.ndimage.interpolation import shift
from scipy.ndimage.interpolation import rotate
from scipy.optimize import curve_fit

import inspect
import trimesh
import image
import GUI
import re
import scipy.io as sio
import pickle
import pywt

try:
    import multiprocessing
except:
    pass
try:
    import cv2
except:
    pass
try:
    import SimpleITK as sitk
except:
    pass
'''
default variables
'''
DEFAULT_DRAW_SIZE=5
WEIGHT_LENGTH_ORDER=1.5



'''
curve fit
'''
def fseries(x,a0,a1,b1,a2,b2,a3,b3):
    period=0.48
    return a0+a1*np.cos(2*np.pi*x/period)+b1*np.sin(2*np.pi*x/period)+a2*np.cos(2.*2*np.pi*x/period)+b2*np.sin(2.*2*np.pi*x/period)+a3*np.cos(3.*2*np.pi*x/period)+b3*np.sin(3.*2*np.pi*x/period)
def timestepPoly(n,timeStepNo):
    return float(n-1)*float(timeStepNo-1-n)/((timeStepNo-1)/2.)**2.



'''
optimisation class
'''

class gradient_ascent:
    def __init__(self,func,initPara,args=(),gain=None,errThreshold=0.6,limitRun=100,maxPara=None,minPara=None,finetune_space=2):
        self.func=func
        self.para=np.array(initPara)
        self.args=args
        self.paraLength=len(initPara)
        if type(errThreshold) in [int,float]:
            self.errThreshold=np.ones(len(initPara))*errThreshold
        self.limitRun=limitRun
        self.slope=1.
        self.finetune_space=finetune_space
        self.fVal=0.
        if maxPara is None:
            maxPara=np.array([float('inf')]*self.paraLength)
        if minPara is None:
            minPara=np.array([float('-inf')]*self.paraLength)
        self.maxPara=maxPara
        self.minPara=minPara
        self.gain=gain
    def run(self,report=float('inf')):
        if report==True:
            report=1
        '''set gain'''
        if self.gain is None:
            self.gain=0.
            gradient=self.grad()
            for n in range(len(gradient)):
                if gradient[n]!=0:
                    if self.para[n]!=0.:
                        self.gain=np.maximum(self.gain,np.abs(self.para[n]/5/gradient[n]))
                    elif self.maxPara[n]<float('inf'):
                        self.gain=np.maximum(self.gain,np.abs(self.maxPara[n]/5/gradient[n]))
                    else:
                        self.gain=np.maximum(self.gain,1)
            if self.gain==0:
                return self.para
        
        error=float('inf')
        self.fVal=self.func(self.para,*self.args)
        if report<float('inf'):
            print('Initial value=',self.fVal,', with',self.para)
        for count in range(1,self.limitRun):
            if error<1.:
                break
            #fValtemp=func(self.para,*self.args)
            gradient=self.grad()
            newPara=self.para+self.gain*self.slope*gradient
            '''reduce gain'''
            for n in range(len(self.para)):
                if newPara[n] > self.maxPara[n]:
                    self.gain=min(self.gain,np.abs((self.maxPara[n]-self.para[n])/self.slope/gradient[n]))
                elif newPara[n] < self.minPara[n]:
                    self.gain=min(self.gain,np.abs((self.minPara[n]-self.para[n])/self.slope/gradient[n]))
            
            newPara=self.para+self.gain*self.slope*gradient
            newfVal=self.func(newPara,*self.args)
            '''reduce gain is fVal did not improve'''
            while ((newfVal>self.fVal) ^ (self.slope==1)):
                self.gain*=0.7
                newPara=self.para+self.gain*self.slope*gradient
                newfVal=self.func(newPara,*self.args)
                if newfVal==self.fVal:
                    break
                if ((newfVal>self.fVal) ^ (self.slope==1)) and np.max(np.abs(newPara-self.para)/self.errThreshold)<1.:
                    newPara=self.para.copy()
                    break
            '''calculate error and update'''
            error=np.max(np.abs(newPara-self.para)/self.errThreshold)
            self.para=np.copy(newPara)
            #print('iteration',count,', parameters=',self.para)
            self.fVal=newfVal
            if count%report==0:
                print('iteration',count,', value=',self.fVal)
        '''fine tune adjustment by errThreshold'''
        print('fine tuning')
        gradient=self.grad()
        for count in range(1,self.limitRun):
            newfVal=[]
            newPara=[]
            for n in range(len(gradient)):
                for m in range(1,self.finetune_space+1):
                    newPara.append(np.copy(self.para))
                    if gradient[n]<0:
                        newPara[-1][n]-=self.errThreshold[n]*self.slope*m
                    elif gradient[n]>0:
                        newPara[-1][n]+=self.errThreshold[n]*self.slope*m
                    newfVal.append(self.func(newPara[-1],*self.args))
            tryIndex=np.argmax(np.array(newfVal))
            if (newfVal[tryIndex]*self.slope)>(self.fVal*self.slope) and np.all(np.array(newPara[tryIndex])<=np.array(self.maxPara)) and np.all(np.array(newPara[tryIndex])>=np.array(self.minPara)):
                self.fVal=newfVal[tryIndex]
                self.para=np.copy(newPara[tryIndex])
                #print('Fine adjustment, parameters=',self.para)
                gradient=self.grad()
            else:
                break
        if report<float('inf'):
            print('Final value=',self.fVal,', with',self.para)
        return np.copy(self.para)

    def grad(self):
        gradient=[]
        for n in range(self.paraLength):
            newPara=np.copy(self.para)
            newPara[n]+=self.errThreshold[n]
            plus=self.func(newPara,*self.args)
            newPara=np.copy(self.para)
            newPara[n]-=self.errThreshold[n]
            minus=self.func(newPara,*self.args)
            gradient.append((plus-minus)/2./self.errThreshold[n])
        return np.array(gradient)
    
class gradient_descent(gradient_ascent):
    def __init__(self,func,initPara,args=(),gain=None,errThreshold=1.,limitRun=100,maxPara=None,minPara=None):
        super(gradient_descent, self).__init__(func,initPara,args=args,gain=gain,errThreshold=errThreshold,limitRun=limitRun,maxPara=maxPara,minPara=minPara)
        self.slope=-1.
'''
internal functions
'''
def calculate_correlation(imageAArray,imageBArray,mask=None):
    '''Calculate the correlation of two images with variance of intensity'''
    imageAArray=np.copy(imageAArray)
    imageBArray=np.copy(imageBArray)
    meanA=imageAArray.mean()
    meanB=imageBArray.mean()
    stdA=np.std(imageAArray)
    stdB=np.std(imageBArray)
    maskVal=0.
    if type(mask)!=type(None):
        sliceList=[]
        for n in range(len(mask)):
            sliceList.append(slice(mask[n][0],mask[n][1]))
        maskVal=np.sum((imageAArray[sliceList]-meanA)*(imageBArray[sliceList]-meanB))
    correl_val=(np.sum((imageAArray-meanA)*(imageBArray-meanB))-maskVal)/stdA/stdB
    return correl_val
    
def translateArray(oldArray,translateLastaxes,includeRotate,fill):
    trlen=len(translateLastaxes)
    if includeRotate:
        for n in range(trlen):
            trlen-=n
            if trlen==(n+1):
                break
    translateIndex=np.zeros(len(oldArray.shape)-trlen)
    translateIndex=np.hstack((translateIndex,translateLastaxes[:trlen]))
    sanitized_translateIndex=[]
    for num in translateIndex:
        sanitized_translateIndex.append(int(np.around(num)))
    newArray=shift(oldArray,sanitized_translateIndex,cval=fill)
    axis=[len(oldArray.shape)-1,len(oldArray.shape)-2]
    for n in range(trlen,len(translateLastaxes)):
        newArray=rotate(newArray,translateLastaxes[n],axes=axis,cval=fill,reshape=False)
        if (axis[0]-1)<=axis[1]:
            axis[1]-=1
            axis[0]=len(oldArray.shape)-1
        else:
            axis[0]-=1
    return newArray
def correlationFunc_translate(translateLastaxes,arrayA,arrayB,includeRotate=False,fill=0,mask=None):
    '''
    mask=[(min1stdim,max1stdim),(min2nddim,max2nddim)...]
    '''
    newArrayB=translateArray(arrayB,translateLastaxes,includeRotate,fill)
    corr=calculate_correlation(arrayA,newArrayB,mask=mask)
    return corr
    
def correlation_translate(arrayA,arrayB,translateLimit,initialTranslate=None,includeRotate=False,fill=0,mask=None):
    '''
    find the translation of arrayB of last [number=len(translateLimit)] of dimension
    that optimises the correlation between arrayA and arrayB
    shift limit limits the maximum shift as a percentage if the shape of array
    '''
    if type(initialTranslate)==type(None):
        iniTrlen=len(translateLimit)
        if includeRotate:
            for n in range(len(translateLimit)):
                iniTrlen+=n
        initialTranslate=np.zeros(iniTrlen)
        
    maxTranslate=np.array(arrayB.shape[-len(translateLimit):])*translateLimit
    maxTranslate=np.concatenate((maxTranslate,10.*np.ones(len(initialTranslate)-len(maxTranslate))))
    minTranslate=-maxTranslate
    optimizing_algorithm=gradient_ascent(correlationFunc_translate,initialTranslate,args=(arrayA,arrayB,includeRotate,fill,mask))
    optimizing_algorithm.maxPara=np.concatenate((maxTranslate,10.*np.ones(len(initialTranslate)-len(maxTranslate))))
    optimizing_algorithm.minPara=minTranslate
    optimizing_algorithm.errThreshold[len(translateLimit):]=np.ones(len(optimizing_algorithm.errThreshold)-len(translateLimit))*1.5
    optimized_translate=optimizing_algorithm.run()
    return optimized_translate

def newPoint_intensityFunc(vecNormLength,imageArray,point1,point2,trueScale,midpoint,unitVecSearch,ori_I,ori_length,lengthWeight):
    '''calculate the line intensity. vecNormLength=[]'''
    moveVec=unitVecSearch*vecNormLength[0]
    newPointA=midpoint+moveVec
    A1_I,A1_length=calIntensity(imageArray,newPointA,point1,trueScale)
    A2_I,A2_length=calIntensity(imageArray,newPointA,point2,trueScale)
    IdensityRatio=(A1_I*A1_length+A2_I*A2_length)/(A1_length+A2_length)/ori_I
    lengthRatio=np.linalg.norm(moveVec*trueScale)/ori_length
    result=IdensityRatio+lengthWeight*lengthRatio**WEIGHT_LENGTH_ORDER
    return result
def pointToIndexTuple(point,shape):
    pointIndex=[]
    for m in range(len(point)):
        tempIndex=int(np.round(point[m]))
        tempIndex=min(max(tempIndex,0),(shape[m]-1))
        pointIndex.append(tempIndex)
    return tuple(pointIndex)
def calIntensity(imageArray,point1,point2,trueScale,step=1.):
    '''point1 is included in calculation'''
    vec=point2-point1
    length=np.linalg.norm(vec)
    unitVec=vec/length*step
    totalIntensity=float(imageArray[pointToIndexTuple(point1,imageArray.shape)])
    count=1
    for n in range(1,int(length/step)):
        totalIntensity+=imageArray[pointToIndexTuple(point1+unitVec*n,imageArray.shape)]
        count+=1
    intensityDensity=totalIntensity/count
    trueLength=np.linalg.norm(vec*trueScale)
    return (intensityDensity,trueLength)
def get_new_midPoint(imageArray,point1,point2,trueScale,lengthWeight):
    midpoint=(point1+point2)/2.
    ori_I,ori_length=calIntensity(imageArray,point1,point2,trueScale)
    if ori_length==0.:
        return midpoint
    if ori_I==0:
        ori_I=1
    perpendVec=np.zeros(len(point1))
    last2vec=(point2-point1)[-2:]
    '''comparing last 2 vec norm with the rest'''
    fullvec_true=(point2-point1)*trueScale
    last2vec_true=fullvec_true[-2:]
    fistfewvec_true=fullvec_true[:-2]
    if np.linalg.norm(last2vec_true)>np.linalg.norm(fistfewvec_true):
        perpendVec[-2:]=last2vec[[1,0]]
        perpendVec[-1]=-perpendVec[-1]
    else:
        perpendVec[-2:]=last2vec
    if np.linalg.norm(perpendVec)==0:
        return midpoint
    perpendVec_normal=perpendVec/np.linalg.norm(perpendVec)
    inputArgs=(imageArray,point1,point2,trueScale,midpoint,perpendVec_normal,ori_I,ori_length,lengthWeight)
    inplaneLength=ori_length*2./(trueScale[-2]+trueScale[-1])
    #optimising=gradient_descent(newPoint_intensityFunc,[0.],args=inputArgs,maxPara=[inplaneLength/2.],minPara=[-inplaneLength/2.])
    optimising=gradient_descent(newPoint_intensityFunc,[0.],args=inputArgs,errThreshold=5.)
    veclength=optimising.run()
    maxcoord=np.array(imageArray.shape)-1
    mincoord=np.zeros(len(imageArray.shape))
    newPoint=midpoint+veclength*perpendVec_normal
    newPoint=np.maximum(np.minimum(newPoint,maxcoord),mincoord)
    return newPoint

def recursive_midpoint_gen(imageArray,trueScale,pointList,subdivide,lengthWeight):
    if type(pointList)==list:
        newPointList=[]
        for n in range(len(pointList)):
            newPointList.append(recursive_midpoint_gen(imageArray,trueScale,pointList[n],subdivide[1:],lengthWeight))
        newPointList=np.array(newPointList)
        dividePointList=[]
        for n in range(len(newPointList[0])):
            dividePointList.append(recursive_midpoint_gen(imageArray,trueScale,newPointList[:,n],subdivide,lengthWeight))
        dividePointList=np.array(dividePointList)
        returnPointList=dividePointList[:,0]
        for n in range(1,len(dividePointList[0])):
            returnPointList=np.vstack((returnPointList,dividePointList[:,n]))
        return returnPointList
    else:
        newPointList=[pointList[0]]
        for n in range(len(pointList)-1):
            newPointList.append(get_new_midPoint(imageArray,pointList[n],pointList[n+1],trueScale,lengthWeight))
            newPointList.append(pointList[n+1])
        newPointList=np.array(newPointList)
        if subdivide[0]>1:
            tempSubdivide=subdivide[:]
            tempSubdivide[0]-=1
            newPointList=recursive_midpoint_gen(imageArray,trueScale,newPointList,tempSubdivide,lengthWeight)
        return newPointList
#####In process of coding parallel
def midpoint_gen_parallel(imageArray,trueScale,pointList,subdivide,lengthWeight,cpucores=0):
    if cpucores==0:
        cpucores=multiprocessing.cpu_count()
    processes=[]
    out_queue=multiprocessing.Queue()
    for n in range(len(pointList)):
        processes.append(multiprocessing.Process(target=parallelCompatibleFunc,args=(n,out_queue,recursive_midpoint_gen),kwargs={'FuncArgs':(imageArray,trueScale,pointList[n],subdivide[1:],lengthWeight)}))
    startProcesses(processes,cpucores)
    
    newPointList=[None]*len(pointList)
    for n in range(len(pointList)):
        result=out_queue.get()
        newPointList[result[0]]=np.copy(result[1])
    newPointList=np.array(newPointList)

    processes=[]
    out_queue=multiprocessing.Queue()
    for n in range(len(newPointList[0])):
        processes.append(multiprocessing.Process(target=parallelCompatibleFunc,args=(n,out_queue,recursive_midpoint_gen),kwargs={'FuncArgs':(imageArray,trueScale,newPointList[:,n],subdivide,lengthWeight)}))
    startProcesses(processes,cpucores)

    dividePointListTemp=[None]*len(newPointList[0])
    for n in range(len(newPointList[0])):
        result=out_queue.get()
        dividePointListTemp[result[0]]=np.copy(result[1])
        
    dividePointList=dividePointListTemp[0]
    for n in range(1,len(newPointList[0])):
        dividePointList=np.vstack((dividePointList,dividePointListTemp[n]))
    return dividePointList
    
def arrangePoints(points,ignoreLast=2,axisNum=0,coordRef=[]):
    '''
    arrange the points ignoring last few axis
    assuming the point are in incremental values according to the axis
    '''
    newList=[]
    lastIndex=0
    index=points[0][axisNum]
    for n in range(1,len(points)):
        if points[n][axisNum]!=index:
            if n-lastIndex==1:
                newList.append([np.copy(points[lastIndex:n])])
            else:
                newList.append(np.copy(points[lastIndex:n]))
            lastIndex=n
            coordRef.append([index])
            index=points[n][axisNum]
    if len(points)-lastIndex==1:
        newList.append([np.copy(points[lastIndex:])])
    else:
        newList.append(np.copy(points[lastIndex:]))
    coordRef.append([index])
    if (axisNum+1)==(len(points[0])-ignoreLast):
        return newList
    for n in range(len(newList)):
        if len(newList[n])>1:
            storeIndex=coordRef[n][-1]
            coordRef[n]=[]
            newList[n]=arrangePoints(newList[n],ignoreLast,axisNum+1,coordRef=coordRef[n])
            coordRef[n].append(storeIndex)
    return newList

def createGaussianMat(size,sigma=1., mu=0.):
    '''size indicate the length from the middle point'''
    lineSpace=[]
    for length in size:
        lineSpace.append(np.linspace(-1,1,length*2+1))
    x= np.meshgrid(*lineSpace)
    x=np.array([*x])
    d=np.sum(x*x,axis=0)
    d=np.sqrt(d)
    gauss = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return gauss

def createPointRangeSlice(point,size,totalIndexLength):
    pointSlice=[]
    drawSlice=[]
    for n in range(len(point)):
        index=int(np.around(point[n]))
        indexMax=int(min(index+size[n]+1,totalIndexLength[n]))
        indexMin=int(max(index-size[n],0))
        pointSlice.append(slice(indexMin,indexMax))
        drawSlice.append(slice(int(indexMin+size[n]-index),int(indexMax+size[n]-index)))
    return (pointSlice,drawSlice)
def pointCenteredFilter(image,pointsList,filterMat):
    image=image.clone()
    matSize=(np.array(filterMat.shape)-1)/2
    for n in range(len(pointsList)):
        pointSlice,drawSlice=createPointRangeSlice(pointsList[n],matSize,image.data.shape)
        tempMat=image.data[pointSlice]*filterMat[drawSlice]
        image.data[pointSlice]=tempMat
    return image

def cv2fill3D(imageArrayShape,arangedPoints3D,maxfill=0,minfill=1):
    returnArray=np.ones(imageArrayShape[-3:])*minfill
    for n in range(len(arangedPoints3D)-1):
        zCoord=(arangedPoints3D[n][0][-3]+arangedPoints3D[n+1][0][-3])/2.
        tempPts=np.fliplr(np.vstack((arangedPoints3D[n][:,-2:],np.flipud(arangedPoints3D[n+1][:,-2:]))))
        zceil=int(np.ceil(zCoord))
        zfloor=int(np.floor(zCoord))
        if zceil==zfloor:
            tempArray=np.copy(returnArray[zceil])
            tempArray=cv2.fillPoly(tempArray,np.array([tempPts],np.int32),0)
            returnArray[zceil]=np.minimum(returnArray[zceil],tempArray)
        else:
            fillceil=maxfill+np.abs(zceil-zCoord)*(minfill-maxfill)
            tempArray=np.copy(returnArray[zceil])
            tempArray=cv2.fillPoly(tempArray,np.array([tempPts],np.int32),fillceil)
            returnArray[zceil]=np.minimum(returnArray[zceil],tempArray)
            fillfloor=maxfill+np.abs(zfloor-zCoord)*(minfill-maxfill)
            tempArray=np.copy(returnArray[zfloor])
            tempArray=cv2.fillPoly(tempArray,np.array([tempPts],np.int32),fillfloor)
            returnArray[zfloor]=np.minimum(returnArray[zfloor],tempArray)
    return returnArray

def pointsToPlanes(pointList,coordRef,numAxisToSnap=1):
    range1=coordRef[0][-1]
    range2=coordRef[-2][-1]
    maxIndex=int(np.ceil(max(range1,range2)))+1
    newList=[[]]*maxIndex
    newCoordRef=[[]]*maxIndex
    trackError=[1.]*maxIndex
    for n in range(len(pointList)):
        nearestIndex=int(np.around(coordRef[n][-1]))
        error=np.abs(coordRef[n][-1]-nearestIndex)
        if error<trackError[nearestIndex]:
            newList[nearealignAxes_translatestIndex]=pointList[n]
            newCoordRef[nearestIndex]=coordRef[n]
            trackError[nearestIndex]=error
    if numAxisToSnap>1:
        for n in range(len(newList)):
            if type(newList[n])!=np.ndarray and newList[n]:
                newList[n],newCoordRef[n]=pointsToPlanes(newList[n],newCoordRef[n],numAxisToSnap=numAxisToSnap-1)
    return (newList,newCoordRef)
def fillplanesWithPoints(points):
    newPoints=[]
    if type(points[0])==np.ndarray:
        lastPlaneCoord=points[0][0][-3]
        newPoints.append(points[0])
        for n in range(1,len(points)):
            nextPlaneCoord=np.around(points[n][0][-3])
            addNum=int(np.abs(nextPlaneCoord-lastPlaneCoord))
            vec=points[n]-points[n-1]
            for m in range(1,addNum+1):
                newPoints.append(points[n-1]+m/float(addNum+1)*vec)
            newPoints.append(points[n])
    else:
        for n in range(len(points)):
            newPoints.append(fillplanesWithPoints(points[n]))
    return newPoints
    
def collectPointsNearestToIndex(pointsList):
    coordRef=[]
    points=arrangePoints(pointsList,coordRef=coordRef)
    coordRef.append(0)
    newPoints,newCoordRefalignAxes_translate=pointsToPlanes(points,coordRef)
    newPoints=fillplanesWithPoints(newPoints)
    return newPoints

'''
external use functions
'''
def alignAxes_translate(image,axesToTranslate,refAxis,dimSlice={},fixedRef=False,includeRotate=False,mask=None):
    '''refAxis={'axis':index}'''
    image=image.clone()
    returnDim=image.dim[:]
    axisref=list(refAxis.keys())[0]
    indexref=refAxis[axisref]
    '''sanitize dimSlice to remove duplicate in axesToTranslate and refAxis'''
    axisSlice=slice(image.data.shape[image.dim.index(axisref)])
    if axisref in dimSlice:
        axisSlice=dimSlice.pop(axisref)
    for axis in axesToTranslate:
        if axis in dimSlice:
            dimSlice.pop(axis)
    for axis in dimSlice:
        if dimSlice[axis]==-1:
            dimSlice[axis]=slice(image.data.shape[image.dim.index(axis)])
    dimensionkey=[axisref]
    dimensionSlice=[axisSlice]
    for dimension in dimSlice:
        dimensionkey.append(dimension)
        dimensionSlice.append(dimSlice[dimension])
    image.rearrangeDim(dimensionkey,True)
    image.rearrangeDim(axesToTranslate,False)

    extractArray=np.copy(image.data[dimensionSlice])
    if axisSlice.start is None:
        axisSliceStart=0
    else:
        axisSliceStart=axisSlice.start
    if axisSlice.step is None:
        axisSliceStep=1
    else:
        axisSliceStep=axisSlice.step   
    relativeIndexref=int((indexref-axisSliceStart)/axisSliceStep)
    translateIndex=None
    saveTranslateIndex=[]
    for n in range(relativeIndexref,0,-1):
        print('running slice',n)
        if fixedRef:
            ref=relativeIndexref
            indSlice=slice(n-1,n)
        else:
            ref=n
            indSlice=slice(0,n)
            translateIndex=None
        translateIndex=correlation_translate(extractArray[ref],extractArray[n-1],np.ones(len(axesToTranslate))*0.5,initialTranslate=translateIndex,includeRotate=includeRotate,mask=mask)
        if (np.abs(translateIndex)>=0.5).any():
            print('updating... with translation',translateIndex)
            extractArray[indSlice]=translateArray(extractArray[indSlice],translateIndex,includeRotate,0)
            saveTranslateIndex.append([n-1,*translateIndex])
    translateIndex=None
    for n in range(relativeIndexref,len(extractArray)-1):
        print('running slice',n)
        if fixedRef:
            ref=relativeIndexref
            indSlice=slice(n+1,n+2)
        else:
            ref=n
            indSlice=slice(n+1,None)
            translateIndex=None
        translateIndex=correlation_translate(extractArray[ref],extractArray[n+1],np.ones(len(axesToTranslate))*0.5,initialTranslate=translateIndex,includeRotate=includeRotate,mask=mask)
        if (np.abs(translateIndex)>=0.5).any():
            print('updating... with translation',translateIndex)
            extractArray[indSlice]=translateArray(extractArray[indSlice],translateIndex,includeRotate,0)
            saveTranslateIndex.append([n+1,*translateIndex])
    
    image.data[dimensionSlice]=np.copy(extractArray)
    image.rearrangeDim(returnDim,True)
    return (image,np.array(saveTranslateIndex))

def snapDraw_black(image,axesToSearch,subdivide,drawSize={},lengthWeight=0.7):
    image=image.clone()
    returnDim=image.dim[:]
    
    image.rearrangeDim(axesToSearch,False)
    
    '''calculate true scale'''
    trueScale=[]
    for dimension in image.dim:
        trueScale.append(image.dimlen[dimension])
    trueScale=np.array(trueScale)

    '''repeatedly user-input and draw'''
    addAnother=True
    allPointList=[]
    while addAnother:
        checkKey=False
        while not(checkKey):
            pointsList=[]
            points=image.readPoints(addInstruct='Exit without clicking will end session.')
            if len(points)==0:
                addAnother=False
                break
            try:
                pointsList=arrangePoints(points)
                pointsList=recursive_midpoint_gen(image.data,trueScale,pointsList,subdivide,lengthWeight)
            except:
                pointsList=[]
                print('Calculation error, Input points again.')
                continue
            checkKey=GUI.image2DGUI(image,initPointList=pointsList,disable=['click'],addInstruct='Press Esc to discard changes.')
            checkKey=checkKey.enter
        if len(points)!=0:
            allPointList.append(pointsList)
            '''draw'''
            #image=drawpoints(image,pointsList,drawSize=drawSize)
            image=drawplanes(image,[pointsList])

    transposeIndex=image.rearrangeDim(axesToSearch,False)
    for n in range(len(allPointList)):
        allPointList[n]=np.array(allPointList[n])
        allPointList[n]=allPointList[n][:,transposeIndex]
    return image,allPointList

def drawpoints(image,allPointList,drawSize={},drawMat=None,onEmpty=False,onBlack=True):
    image=image.clone()
    if onEmpty:
        image.empty()
        image.invert()
    if drawMat is None:
        '''create draw multiplier matrix'''
        for dimension in image.dim:
            if dimension not in drawSize:
                drawSize[dimension]=DEFAULT_DRAW_SIZE
        drawMatSize=[]
        for dimension in image.dim:
            drawMatSize.append(drawSize[dimension])
        drawMat=1.-createGaussianMat(drawMatSize)
    image=pointCenteredFilter(image,allPointList,drawMat)
    if onEmpty:
        if onBlack:
            image.invert()
        image.show()
    return image
def drawplanes(image,planePointList,onEmpty=False,onBlack=True):
    image=image.clone()
    if onEmpty:
        image.empty()
        image.invert()
    for n in range(len(planePointList)):
        newPoints=collectPointsNearestToIndex(planePointList[n])
        for t in range(len(newPoints)):
            if newPoints[t]:
                mask=cv2fill3D(image.data.shape,newPoints[t])
                image.data[t]=image.data[t]*mask
    if onEmpty:
        if onBlack:
            image.invert()
        image.show()
    return image

def convertElastxOutputToNumpy(file): #return xyz format
    with open (file, "r") as myfile:
        data=myfile.readlines()
    arrayNew=[]
    for string in data:
        result = re.search('OutputPoint(.*)Deformation', string)
        arrayNew.append(np.fromstring(result.group(1)[5:-6], sep=' '))
    arrayNew=np.array(arrayNew)
    return arrayNew

def imageRegister(image,stlFile,imageToSTLsize=[],stlRefDim={},baseRefFraction=1.,baseRefFunc=None,verifyImg=False):
    image=image.clone()#require default dimension
        
    if not stlRefDim:
        stlRefDim[image.dim[0]]=0
    if not imageToSTLsize:
        for dimension in image.dim[-3:]:
            imageToSTLsize.insert(0,image.dimlen[dimension])
        
    imageToSTLsize=np.array(imageToSTLsize)

    '''adjust imageToSTLsize'''
    
    '''read STL'''
    ref_mesh=trimesh.load(stlFile)
    vertices=np.array(ref_mesh.vertices)
    pointfile=stlFile[:-4]
    pixelVertice=vertices/imageToSTLsize
    os.makedirs(pointfile+'Output', exist_ok=True)
    np.savetxt(pointfile+'Output/outputpoints.pts',pixelVertice,header='point\n'+str(len(pixelVertice)),comments='')
    np.savetxt(pointfile+'Output/outputpoints0.pts',pixelVertice,header='point\n'+str(len(pixelVertice)),comments='')

    parameterMapVector = sitk.VectorOfParameterMap()
    affine=sitk.GetDefaultParameterMap("affine")
    parameterMapVector.append(affine)
    bspline=sitk.GetDefaultParameterMap("bspline")
    #bspline["Metric"]=(*bspline["Metric"],"DisplacementMagnitudePenalty")
    parameterMapVector.append(bspline)
    
    '''start'''
    for n in range(image.data.shape[0]-1):
        if baseRefFunc is not None:
            baseRefFraction=baseRefFunc(float(n+1)/image.data.shape[0])
        if baseRefFraction!=1. and  n!=0:
            print('Registering t',n+1,' wrt t',n)
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOn()
            elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(np.copy(image.data[n]), isVector=False))
            elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=False))
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            if verifyImg:
                sitk.WriteImage(elastixImageFilter.GetResultImage(),pointfile+'Output/resultImgnb'+str(n+1)+'.mha')
            print('Transforming t',n,' to t',n+1)
            transformixImageFilter=sitk.TransformixImageFilter()
            transformixImageFilter.LogToFileOn()
            transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
            transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=False))
            transformixImageFilter.SetFixedPointSetFileName(pointfile+'Output/outputpoints.pts')
            transformixImageFilter.SetOutputDirectory(pointfile+'Output')
            transformixImageFilter.Execute()
            newPixelVertice_neighbour=convertElastxOutputToNumpy(pointfile+'Output/outputpoints.txt')
        if baseRefFraction!=0. or n==0:
            print('Registering t',n+1,' wrt t',0)
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOn()
            elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(np.copy(image.data[0]), isVector=False))
            elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=False))
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            if verifyImg:
                sitk.WriteImage(elastixImageFilter.GetResultImage(),pointfile+'Output/resultImgbase'+str(n+1)+'.mha')
            print('Transforming t',0,' to t',n+1)
            transformixImageFilter=sitk.TransformixImageFilter()
            transformixImageFilter.LogToFileOn()
            transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
            transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=False))
            transformixImageFilter.SetFixedPointSetFileName(pointfile+'Output/outputpoints0.pts')
            transformixImageFilter.SetOutputDirectory(pointfile+'Output')
            transformixImageFilter.Execute()
            newPixelVertice_base=convertElastxOutputToNumpy(pointfile+'Output/outputpoints.txt')
        if baseRefFraction==1. or n==0:
            newPixelVertice=newPixelVertice_base
        elif baseRefFraction==0.:
            newPixelVertice=newPixelVertice_neighbour
        else:
            newPixelVertice=newPixelVertice_base*baseRefFraction+newPixelVertice_neighbour*(1.-baseRefFraction)
        '''write stl'''
        print('Writing STL file to ',pointfile+'_t'+str(n+1)+'.stl')
        
        ref_mesh.vertices=newPixelVertice*imageToSTLsize
        trimesh.io.export.export_mesh(ref_mesh,pointfile+'_t'+str(n+1)+'.stl')
        if baseRefFraction!=1.:
            np.savetxt(pointfile+'Output/outputpoints.pts',newPixelVertice,header='point\n'+str(len(pixelVertice)),comments='')
def vectorRegister(image,savePath='',stlRefDim={},baseRefFraction=1.,baseRefFunc=None,verifyImg=False,startind=0):
    image=image.clone()
    if not stlRefDim:
        stlRefDim[image.dim[0]]=0
    if not(savePath):
        savePath='registeredVector.mat'
    '''adjust imageToSTLsize'''
    
    '''setup points'''
    gridshape=[]
    for n in range(-1,-len(image.data.shape),-1):
        gridshape.insert(0,slice(image.data.shape[n]))
    x=np.mgrid[gridshape]
    pixelVertice = np.vstack(map(np.ravel, x))
    pixelVertice=np.fliplr(pixelVertice.transpose()) # in xyz format
    pointfile=savePath[:-4]
    os.makedirs(pointfile+'Output', exist_ok=True)
    np.savetxt(pointfile+'Output/outputpoints.pts',pixelVertice,header='point\n'+str(len(pixelVertice)),comments='')
    np.savetxt(pointfile+'Output/outputpoints0.pts',pixelVertice,header='point\n'+str(len(pixelVertice)),comments='')

    parameterMapVector = sitk.VectorOfParameterMap()
    affine=sitk.GetDefaultParameterMap("affine")
    parameterMapVector.append(affine)
    bspline=sitk.GetDefaultParameterMap("bspline")
    #bspline["Metric"]=(*bspline["Metric"],"DisplacementMagnitudePenalty")
    parameterMapVector.append(bspline)
    
    '''start'''
    colorVec=False
    if 'RGB' in image.dim:
        colorVec=True
    for n in range(startind,image.data.shape[0]-1):
        if baseRefFunc is not None:
            baseRefFraction=baseRefFunc(float(n+1)/image.data.shape[0])
        if baseRefFraction!=1. and  n!=startind:
            print('Registering t',n+1,' wrt t',n)
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOn()
            elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(np.copy(image.data[n]), isVector=colorVec))
            elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=colorVec))
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            print('Transforming t',n,' to t',n+1)
            transformixImageFilter=sitk.TransformixImageFilter()
            transformixImageFilter.LogToFileOn()
            transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
            transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=colorVec))
            transformixImageFilter.SetFixedPointSetFileName(pointfile+'Output/outputpoints.pts')
            transformixImageFilter.SetOutputDirectory(pointfile+'Output')
            transformixImageFilter.Execute()
            if verifyImg:
                sitk.WriteImage(elastixImageFilter.GetResultImage(),pointfile+'Output/resultImgnb'+str(n+1)+'.mha')
            newPixelVertice_neighbour=convertElastxOutputToNumpy(pointfile+'Output/outputpoints.txt')
        if baseRefFraction!=0. or n==startind:
            print('Registering t',n+1,' wrt t',0)
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOn()
            elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(np.copy(image.data[0]), isVector=colorVec))
            elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=colorVec))
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            #Tmap=elastixImageFilter.GetTransformParameterMap()
            #Tmap[0]["TransformParameters"]
            if verifyImg:
                sitk.WriteImage(elastixImageFilter.GetResultImage(),pointfile+'Output/resultImgbase'+str(n+1)+'.mha')
            print('Transforming t',0,' to t',n+1)
            transformixImageFilter=sitk.TransformixImageFilter()
            transformixImageFilter.LogToFileOn()
            transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
            transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=colorVec))
            transformixImageFilter.SetFixedPointSetFileName(pointfile+'Output/outputpoints0.pts')
            transformixImageFilter.SetOutputDirectory(pointfile+'Output')
            transformixImageFilter.Execute()
            newPixelVertice_base=convertElastxOutputToNumpy(pointfile+'Output/outputpoints.txt')
        if baseRefFraction==1. or n==startind:
            newPixelVertice=newPixelVertice_base
        elif baseRefFraction==0.:
            newPixelVertice=newPixelVertice_neighbour
        else:
            newPixelVertice=newPixelVertice_base*baseRefFraction+newPixelVertice_neighbour*(1.-baseRefFraction)
        
        vector=newPixelVertice-pixelVertice #xyz format
        vectorMap=np.fliplr(vector).reshape((*image.data.shape[1:],len(image.data.shape[1:])))
        print('Writing MAT file to ',pointfile+str(n+1)+'.mat')
        sio.savemat(pointfile+str(n+1)+'.mat',{('registeredVector'+str(n+1)):vectorMap})
        if baseRefFraction!=1.:
            np.savetxt(pointfile+'Output/outputpoints.pts',newPixelVertice,header='point\n'+str(len(pixelVertice)),comments='')

def TmapRegister(image,savePath='',origin=(0.,0.,0.),bgrid=2.,bweight=1.,rms=False,startTime=0,scaleImg=1.,writeImg=False,twoD=False,nres =3,smoothing=True):
    image=image.clone()
    if scaleImg!=1:
        for axis in ['x','y','z']:
            if axis in image.dimlen:
                image.dimlen[axis]=image.dimlen[axis]*scaleImg
    if not(savePath):
        savePath=os.path.dirname(os.path.realpath(__file__))
    '''adjust imageToSTLsize'''
    
    '''setup points'''
    '''
    gridshape=[]
    for n in range(-1,-len(image.data.shape),-1):
        gridshape.insert(0,slice(image.data.shape[n]))
    x=np.mgrid[gridshape]
    pixelVertice = np.vstack(map(np.ravel, x))
    pixelVertice=np.fliplr(pixelVertice.transpose()) # in xyz format
    pointfile=savePath[:-4]
    os.makedirs(pointfile+'Output', exist_ok=True)
    np.savetxt(pointfile+'Output/outputpoints.pts',pixelVertice,header='point\n'+str(len(pixelVertice)),comments='')
    np.savetxt(pointfile+'Output/outputpoints0.pts',pixelVertice,header='point\n'+str(len(pixelVertice)),comments='')
    '''
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+'/transform', exist_ok=True)
    if twoD:
        spacing=(image.dimlen['x'],image.dimlen['y'])
        if len(origin)>2:
            origin=origin[:2]
    else:
        spacing=(image.dimlen['x'],image.dimlen['y'],image.dimlen['z'])
    parameterMapVector = sitk.VectorOfParameterMap() 
    affine=sitk.GetDefaultParameterMap("affine")
    affine["AutomaticScalesEstimation"]=( "true", )
    affine["AutomaticTransformInitialization"] = ( "true", )
    bspline=sitk.GetDefaultParameterMap("bspline")
    bspline["AutomaticScalesEstimation"]=( "true", )
    bspline["AutomaticTransformInitialization"] = ( "true", )
    bspline['Metric0Weight']=(str(bweight),)
    bspline['FinalGridSpacingInPhysicalUnits']=(str(max(spacing)*bgrid),)
    #bspline["Metric"]=(*bspline["Metric"],"DisplacementMagnitudePenalty")
    if rms:
        bspline["Metric"]=("AdvancedMeanSquares",bspline["Metric"][1])

    #parameterMapVector.append(affine)
    parameterMapVector.append(bspline)
    
    if nres != 3:
        for trans in parameterMapVector:
            trans["NumberOfResolutions"] = (str(nres),)
            if smoothing:
                if twoD:
                    trans["FixedImagePyramidSchedule"] =  tuple(np.repeat(2**np.arange(nres), 2).astype(str))
                    trans["MovingImagePyramidSchedule"] = tuple(np.repeat(2**np.arange(nres), 2).astype(str))
                else:
                    trans["FixedImagePyramidSchedule"] =  tuple(np.repeat(2**np.arange(nres), 3).astype(str))
                    trans["MovingImagePyramidSchedule"] = tuple(np.repeat(2**np.arange(nres), 3).astype(str))
    
    '''start'''
    colorVec=False
    if 'RGB' in image.dim:
        colorVec=True

    fixImg=sitk.GetImageFromArray(np.copy(image.data[0]), isVector=colorVec)
    fixImg.SetOrigin(origin)
    fixImg.SetSpacing(spacing)
    sitk.WriteImage(fixImg,savePath+'/t0Img.mha')
    timeList=np.array(range(len(image.data)))*image.dimlen['t']
    np.savetxt(savePath+'/transform/timeList',timeList)
    for n in range(startTime,image.data.shape[0]-1):
        if n!=0:
            print('Registering t',n+1,' wrt t',n)
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOn()
            elastixImageFilter.LogToConsoleOff()
            fixImg=sitk.GetImageFromArray(np.copy(image.data[n]), isVector=colorVec)
            fixImg.SetOrigin(origin)
            fixImg.SetSpacing(spacing)
            movImg=sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=colorVec)
            movImg.SetOrigin(origin)
            movImg.SetSpacing(spacing)
            elastixImageFilter.SetFixedImage(fixImg)
            elastixImageFilter.SetMovingImage(movImg)
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            Tmap=elastixImageFilter.GetTransformParameterMap()
            for m in range(len(Tmap)):
                sitk.WriteParameterFile(Tmap[m],savePath+'/transform/t'+str(n)+'to'+str(n+1)+'_'+str(m)+'.txt')
            if writeImg:
                sitk.WriteImage(elastixImageFilter.GetResultImage(),savePath+'/t'+str(n)+'to'+str(n+1)+'_resultImg.mha')
        print('Registering t',n+1,' wrt t',0)
        elastixImageFilter=sitk.ElastixImageFilter()
        elastixImageFilter.LogToFileOn()
        elastixImageFilter.LogToConsoleOff()
        fixImg=sitk.GetImageFromArray(np.copy(image.data[0]), isVector=colorVec)
        fixImg.SetOrigin(origin)
        fixImg.SetSpacing(spacing)
        movImg=sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=colorVec)
        movImg.SetOrigin(origin)
        movImg.SetSpacing(spacing)
        elastixImageFilter.SetFixedImage(fixImg)
        elastixImageFilter.SetMovingImage(movImg)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.Execute()
        Tmap=elastixImageFilter.GetTransformParameterMap()
        for m in range(len(Tmap)):
            sitk.WriteParameterFile(Tmap[m],savePath+'/transform/t0to'+str(n+1)+'_'+str(m)+'.txt')
        if writeImg:
            sitk.WriteImage(elastixImageFilter.GetResultImage(),savePath+'/t0to'+str(n+1)+'_resultImg.mha')
def TmapRegister_img2img(image1,image2,savePath='',fileName='img2img',scaleImg=1.,tInd=None,origin1=(0.,0.,0.),origin2=(0.,0.,0.),EulerTransformCorrection=False,rms=False,bgrid=2.,bweight=1.,twoD=False,nres =3,smoothing=True):
    image1=image1.clone()
    image2=image2.clone()
    if scaleImg!=1:
        for axis in ['x','y','z']:
            if axis in image1.dimlen:
                image1.dimlen[axis]=image1.dimlen[axis]*scaleImg
            if axis in image2.dimlen:
                image2.dimlen[axis]=image2.dimlen[axis]*scaleImg
    if not(savePath):
        savePath=os.path.dirname(os.path.realpath(__file__))
    '''adjust imageToSTLsize'''
    
    '''setup points'''
    '''
    gridshape=[]
    for n in range(-1,-len(image.data.shape),-1):
        gridshape.insert(0,slice(image.data.shape[n]))
    x=np.mgrid[gridshape]
    pixelVertice = np.vstack(map(np.ravel, x))
    pixelVertice=np.fliplr(pixelVertice.transpose()) # in xyz format
    pointfile=savePath[:-4]
    os.makedirs(pointfile+'Output', exist_ok=True)
    np.savetxt(pointfile+'Output/outputpoints.pts',pixelVertice,header='point\n'+str(len(pixelVertice)),comments='')
    np.savetxt(pointfile+'Output/outputpoints0.pts',pixelVertice,header='point\n'+str(len(pixelVertice)),comments='')
    '''
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+'/transform', exist_ok=True)
    if twoD:
        spacing1=(image1.dimlen['x'],image1.dimlen['y'])
        spacing2=(image2.dimlen['x'],image2.dimlen['y'])
        if len(origin1)>2:
            origin1=origin1[:2]
        if len(origin2)>2:
            origin2=origin2[:2]
    else:
        spacing1=(image1.dimlen['x'],image1.dimlen['y'],image1.dimlen['z'])
        spacing2=(image2.dimlen['x'],image2.dimlen['y'],image2.dimlen['z'])
    
    parameterMapVector = sitk.VectorOfParameterMap()
    EulerTransform=sitk.GetDefaultParameterMap("rigid")
    EulerTransform["AutomaticScalesEstimation"]=( "true", )
    EulerTransform["AutomaticTransformInitialization"] = ( "true", ) 
    affine=sitk.GetDefaultParameterMap("affine")
    affine["AutomaticScalesEstimation"]=( "true", )
    affine["AutomaticTransformInitialization"] = ( "true", )
    bspline=sitk.GetDefaultParameterMap("bspline")
    bspline["AutomaticScalesEstimation"]=( "true", )
    bspline["AutomaticTransformInitialization"] = ( "true", ) 
    bspline['Metric0Weight']=(str(bweight),)
    bspline['FinalGridSpacingInPhysicalUnits']=(str(max(spacing1)*bgrid),)
    #bspline["Metric"]=(*bspline["Metric"],"DisplacementMagnitudePenalty")
    bspline["Metric"]=("AdvancedMeanSquares",bspline["Metric"][1])
    if rms:
        bspline["Metric"]=("AdvancedMeanSquares",bspline["Metric"][1])
    if EulerTransformCorrection:
        parameterMapVector.append(EulerTransform)
    parameterMapVector.append(bspline)
    
    if nres != 3:
        for trans in parameterMapVector:
            trans["NumberOfResolutions"] = (str(nres),)
            if smoothing:
                if twoD:
                    trans["FixedImagePyramidSchedule"] =  tuple(np.repeat(2**np.arange(nres), 2).astype(str))
                    trans["MovingImagePyramidSchedule"] = tuple(np.repeat(2**np.arange(nres), 2).astype(str))
                else:
                    trans["FixedImagePyramidSchedule"] =  tuple(np.repeat(2**np.arange(nres), 3).astype(str))
                    trans["MovingImagePyramidSchedule"] = tuple(np.repeat(2**np.arange(nres), 3).astype(str))
    
    '''start'''
    colorVec=False
    if 'RGB' in image1.dim:
        colorVec=True
        
    if type(tInd)==type(None):
        fixImg=sitk.GetImageFromArray(np.copy(image1.data), isVector=colorVec)
    else:
        fixImg=sitk.GetImageFromArray(np.copy(image1.data[tInd[0]]), isVector=colorVec)
    fixImg.SetOrigin(origin1)
    fixImg.SetSpacing(spacing1)    
    sitk.WriteImage(fixImg,savePath+'/t0Img.mha')
    if type(tInd)==type(None):
        elastixImageFilter=sitk.ElastixImageFilter()
        elastixImageFilter.LogToFileOn()
        elastixImageFilter.LogToConsoleOff()
        fixImg=sitk.GetImageFromArray(np.copy(image1.data), isVector=colorVec)
        fixImg.SetOrigin(origin1)
        fixImg.SetSpacing(spacing1)
        movImg=sitk.GetImageFromArray(np.copy(image2.data), isVector=colorVec)
        movImg.SetOrigin(origin2)
        movImg.SetSpacing(spacing2)
        elastixImageFilter.SetFixedImage(fixImg)
        elastixImageFilter.SetMovingImage(movImg)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.Execute()
        Tmap=elastixImageFilter.GetTransformParameterMap()
        for m in range(len(Tmap)):
            sitk.WriteParameterFile(Tmap[m],savePath+'/transform/'+fileName+'.txt')
    else:
        for n in tInd:
            print('Registering t',n)
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOn()
            elastixImageFilter.LogToConsoleOff()
            fixImg=sitk.GetImageFromArray(np.copy(image1.data[n]), isVector=colorVec)
            fixImg.SetOrigin(origin1)
            fixImg.SetSpacing(spacing1)
            movImg=sitk.GetImageFromArray(np.copy(image2.data[n]), isVector=colorVec)
            movImg.SetOrigin(origin2)
            movImg.SetSpacing(spacing2)
            elastixImageFilter.SetFixedImage(fixImg)
            elastixImageFilter.SetMovingImage(movImg)
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            Tmap=elastixImageFilter.GetTransformParameterMap()
            for m in range(len(Tmap)):
                sitk.WriteParameterFile(Tmap[m],savePath+'/transform/'+fileName+'_t'+str(n)+'.txt')
def TmapRegister_rigid(image1,image2,savePath='',fileInit=None,fileName='img2img',origin1=(0.,0.,0.),origin2=(0.,0.,0.),bsplineTransformCorrection=False,rms=True,bgrid=2.,bweight=1.):
    image1=image1.clone()
    image2=image2.clone()
    image1.rearrangeDim(['z','y','x'])
    image2.rearrangeDim(['z','y','x'])
    if not(savePath):
        savePath=os.path.dirname(os.path.realpath(__file__))
    '''adjust imageToSTLsize'''
    
    '''setup points'''
    '''
    gridshape=[]
    for n in range(-1,-len(image.data.shape),-1):
        gridshape.insert(0,slice(image.data.shape[n]))
    x=np.mgrid[gridshape]
    pixelVertice = np.vstack(map(np.ravel, x))
    pixelVertice=np.fliplr(pixelVertice.transpose()) # in xyz format
    pointfile=savePath[:-4]
    os.makedirs(pointfile+'Output', exist_ok=True)
    np.savetxt(pointfile+'Output/outputpoints.pts',pixelVertice,header='point\n'+str(len(pixelVertice)),comments='')
    np.savetxt(pointfile+'Output/outputpoints0.pts',pixelVertice,header='point\n'+str(len(pixelVertice)),comments='')
    '''
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+'/transform', exist_ok=True)
    spacing1=(image1.dimlen['x'],image1.dimlen['y'],image1.dimlen['z'])
    spacing2=(image2.dimlen['x'],image2.dimlen['y'],image2.dimlen['z'])
    
    parameterMapVector = sitk.VectorOfParameterMap()
    EulerTransform=sitk.GetDefaultParameterMap("rigid")
    EulerTransform["AutomaticScalesEstimation"]=( "true", )
    EulerTransform["AutomaticTransformInitialization"] = ( "true", ) 
    affine=sitk.GetDefaultParameterMap("affine")
    affine["AutomaticScalesEstimation"]=( "true", )
    affine["AutomaticTransformInitialization"] = ( "true", ) 
    bspline=sitk.GetDefaultParameterMap("bspline")
    bspline["AutomaticScalesEstimation"]=( "true", )
    bspline["AutomaticTransformInitialization"] = ( "true", ) 
    bspline['Metric0Weight']=(str(bweight),)
    bspline['FinalGridSpacingInPhysicalUnits']=(str(max(spacing1)*bgrid),)
    #bspline["Metric"]=(*bspline["Metric"],"DisplacementMagnitudePenalty")
    bspline["Metric"]=("AdvancedMeanSquares",bspline["Metric"][1])
    if rms:
        EulerTransform["Metric"]=("AdvancedMeanSquares",)
        bspline["Metric"]=("AdvancedMeanSquares",bspline["Metric"][1])
    parameterMapVector.append(EulerTransform)
    if bsplineTransformCorrection:
        parameterMapVector.append(bspline)
    
    '''start'''
    colorVec=False
    if 'RGB' in image1.dim:
        colorVec=True
    #if type(origin1)==type(None) or type(origin2)==type(None):
    #    origin1=tuple(-0.5*np.array(spacing1)*np.array([image1.data.shape[2],image1.data.shape[1],image1.data.shape[0]]))
    #    origin2=tuple(origin1)
        
    
    elastixImageFilter=sitk.ElastixImageFilter()
    elastixImageFilter.LogToFileOn()
    elastixImageFilter.LogToConsoleOff()
    fixImg=sitk.GetImageFromArray(np.copy(image1.data), isVector=colorVec)
    fixImg.SetOrigin(origin1)
    fixImg.SetSpacing(spacing1)
    sitk.WriteImage(fixImg,savePath+'/'+fileName+'_fixImg.mha')
    movImg=sitk.GetImageFromArray(np.copy(image2.data), isVector=colorVec)
    movImg.SetOrigin(origin2)
    movImg.SetSpacing(spacing2)
    sitk.WriteImage(movImg,savePath+'/'+fileName+'_movImg.mha')
    elastixImageFilter.SetFixedImage(fixImg)
    elastixImageFilter.SetMovingImage(movImg)
    if type(fileInit)==type(None):
        fileInit=savePath+'/transform/'+fileName+'_init.txt'
    if os.path.isfile(fileInit):
        elastixImageFilter.SetInitialTransformParameterFileName(fileInit)
        print('Set initial transform parameters:',fileInit)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    Tmap=elastixImageFilter.GetTransformParameterMap()
    if os.path.isfile(fileInit):
        for m in range(len(Tmap)):
            sitk.WriteParameterFile(Tmap[m],savePath+'/transform/'+fileName+'_'+str(m)+'.txt')
    else:
        sitk.WriteParameterFile(Tmap[0],fileInit)
    sitk.WriteImage(elastixImageFilter.GetResultImage(),savePath+'/'+fileName+'_resultImg.mha')

def transform(stlFile,timeStepNo,mapNo,startTime=1,cumulative=True,ratioFunc=timestepPoly,savePath='',TmapPath='',scale=1.,delimiter=' '):
    if savePath=='':
        savePath=stlFile[:-4]
    if TmapPath=='':
        TmapPath=savePath
    os.makedirs(savePath, exist_ok=True)
    if stlFile[-3:]=='stl':
        ref_mesh=trimesh.load(stlFile)
        oriPos=np.array(ref_mesh.vertices)/scale
    else:
        oriPos=np.loadtxt(stlFile,delimiter=delimiter)/scale
    np.savetxt(savePath+'/input0.pts',oriPos,header='point\n'+str(len(oriPos)),comments='')
    for n in range(startTime,timeStepNo):
        Tmap=[]#SimpleITK.VectorOfParameterMap()
        for m in range(mapNo):
            Tmap.append(sitk.ReadParameterFile(TmapPath+'/transform/t0to'+str(n)+'_'+str(m)+'.txt'))
        
        transformixImageFilter=sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(Tmap)
        transformixImageFilter.SetMovingImage(sitk.ReadImage(TmapPath+'/t0Img.mha'))
        transformixImageFilter.SetFixedPointSetFileName(savePath+'/input0.pts')
        transformixImageFilter.SetOutputDirectory(savePath)
        transformixImageFilter.Execute()

        with open (savePath+'/outputpoints.txt', "r") as myfile:
            data=myfile.readlines()
        newPos=[]
        for string in data:
            result = re.search('OutputPoint(.*)Deformation', string)
            newPos.append(np.fromstring(result.group(1)[5:-6], sep=' '))
        newPos=np.array(newPos)
        addSaveStr=''
        if cumulative:
            addSaveStr='_cumulative'
            if n>1 and n<(timeStepNo-1):
                ratio=ratioFunc(n,timeStepNo)
                Tmap=[]#SimpleITK.VectorOfParameterMap()
                for m in range(mapNo):
                    Tmap.append(sitk.ReadParameterFile(TmapPath+'/transform/t'+str(n-1)+'to'+str(n)+'_'+str(m)+'.txt'))
                
                transformixImageFilter=sitk.TransformixImageFilter()
                transformixImageFilter.SetTransformParameterMap(Tmap)
                transformixImageFilter.SetMovingImage(sitk.ReadImage(TmapPath+'/t0Img.mha'))
                transformixImageFilter.SetFixedPointSetFileName(savePath+'/input.pts')
                transformixImageFilter.SetOutputDirectory(savePath)
                transformixImageFilter.Execute()

                with open (savePath+'/outputpoints.txt', "r") as myfile:
                    data=myfile.readlines()
                newPos2=[]
                for string in data:
                    result = re.search('OutputPoint(.*)Deformation', string)
                    newPos2.append(np.fromstring(result.group(1)[5:-6], sep=' '))
                newPos=newPos*(1.-ratio)+(ratio)*np.array(newPos2)
            np.savetxt(savePath+'/input.pts',newPos,header='point\n'+str(len(newPos)),comments='')
        if stlFile[-3:]=='stl':
            ref_mesh.vertices=np.array(newPos)*scale
            trimesh.io.export.export_mesh(ref_mesh,savePath+'/t'+str(n)+addSaveStr+'.stl')
        else:
            np.savetxt(savePath+'/t'+str(n)+addSaveStr+'.txt',np.array(newPos)*scale)
def transform_img2img(stlFile,trfFile,savePath='',mhaFile='',fileName='trf',scale=1.,delimiter=' '):
    if savePath=='':
        savePath=stlFile[:-4]
    os.makedirs(savePath, exist_ok=True)
    if stlFile[-3:]=='stl':
        ref_mesh=trimesh.load(stlFile)
        oriPos=np.array(ref_mesh.vertices)/scale
    else:
        oriPos=np.loadtxt(stlFile,delimiter=delimiter)/scale
    np.savetxt(savePath+'/input0.pts',oriPos,header='point\n'+str(len(oriPos)),comments='')

    Tmap=[]#SimpleITK.VectorOfParameterMap()
    Tmap.append(sitk.ReadParameterFile(trfFile))
    
    transformixImageFilter=sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(Tmap)
    if mhaFile=='':
        transformixImageFilter.SetMovingImage(sitk.ReadImage(savePath+'/t0Img.mha'))
    else:
        transformixImageFilter.SetMovingImage(sitk.ReadImage(mhaFile))
    
    transformixImageFilter.SetFixedPointSetFileName(savePath+'/input0.pts')
    transformixImageFilter.SetOutputDirectory(savePath)
    transformixImageFilter.Execute()

    with open (savePath+'/outputpoints.txt', "r") as myfile:
        data=myfile.readlines()
    newPos=[]
    for string in data:
        result = re.search('OutputPoint(.*)Deformation', string)
        newPos.append(np.fromstring(result.group(1)[5:-6], sep=' '))
    newPos=np.array(newPos)
    if stlFile[-3:]=='stl':
        ref_mesh.vertices=np.array(newPos)*scale
        trimesh.io.export.export_mesh(ref_mesh,savePath+'/'+fileName+'.stl')
    else:
        np.savetxt(savePath+'/'+fileName+'.txt',np.array(newPos)*scale)

def fittransform(savePath,timeStepNo,addSaveStr='_cumulative'):
    data=[]
    for n in range(timeStepNo):
        data.append(np.loadtxt(savePath+'/t'+str(n)+addSaveStr+'.txt'))
    data=np.array(data)
    for n in range(len(data[0])):
        popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))


def TmapRegisterGroupwise(image,savePath='',origin=(0.,0.,0.,0.),bweight=1.,bgrid=2.):
    image=image.clone()
    if not(savePath):
        savePath=os.path.dirname(os.path.realpath(__file__))
    '''adjust imageToSTLsize'''
    
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+'/transform', exist_ok=True)
    spacing=(image.dimlen['x'],image.dimlen['y'],image.dimlen['z'],image.dimlen['z'])

    parameterMapVector = sitk.VectorOfParameterMap()
    affine=sitk.GetDefaultParameterMap("affine")
    bspline=sitk.GetDefaultParameterMap("bspline")
    bspline['Metric0Weight']=(str(bweight),)
    bspline['FinalGridSpacingInPhysicalUnits']=(str(max(spacing)*bgrid),)
    #bspline["Metric"]=(*bspline["Metric"],"DisplacementMagnitudePenalty")
    bspline["Metric"]=("AdvancedMeanSquares",bspline["Metric"][1])
    bspline["UseCyclicTransform"]=('true',)
    #parameterMapVector.append(affine)
    parameterMapVector.append(bspline)
    '''start'''
    colorVec=False
    if 'RGB' in image.dim:
        colorVec=True

    fixImg=sitk.GetImageFromArray(np.copy(image.data[0]), isVector=colorVec)
    fixImg.SetOrigin(origin)
    fixImg.SetSpacing(spacing)
    sitk.WriteImage(fixImg,savePath+'/t0Img.mha')
    timeList=np.array(range(len(image.data)))*image.dimlen['t']
    np.savetxt(savePath+'/transform/timeList',timeList)

    vectorOfImages = sitk.VectorOfImage()
    for n in range(image.data.shape[0]):
        vectorOfImages.push_back(sitk.GetImageFromArray(np.copy(image.data[n]), isVector=colorVec))
    imageSITK = sitk.JoinSeries(vectorOfImages)
        
    elastixImageFilter=sitk.ElastixImageFilter()
    elastixImageFilter.LogToFileOn()
    imageSITK.SetOrigin(origin)
    imageSITK.SetSpacing(spacing)
    elastixImageFilter.SetFixedImage(imageSITK)
    elastixImageFilter.SetMovingImage(imageSITK)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    Tmap=elastixImageFilter.GetTransformParameterMap()
    sitk.WriteParameterFile(Tmap[0],savePath+'/transform/groupwise.txt')
            
def inverseRegister(image,refTimeInd,savePath='',origin=(0.,0.,0.),bgrid=2.,bweight=1.,rms=False):
    image=image.clone()
    if not(savePath):
        savePath=os.path.dirname(os.path.realpath(__file__))
    '''adjust imageToSTLsize'''
    
    '''setup points'''
    '''
    gridshape=[]
    for n in range(-1,-len(image.data.shape),-1):
        gridshape.insert(0,slice(image.data.shape[n]))
    x=np.mgrid[gridshape]
    pixelVertice = np.vstack(map(np.ravel, x))
    pixelVertice=np.fliplr(pixelVertice.transpose()) # in xyz format
    pointfile=savePath[:-4]
    os.makedirs(pointfile+'Output', exist_ok=True)
    np.savetxt(pointfile+'Output/outputpoints.pts',pixelVertice,header='point\n'+str(len(pixelVertice)),comments='')
    np.savetxt(pointfile+'Output/outputpoints0.pts',pixelVertice,header='point\n'+str(len(pixelVertice)),comments='')
    '''
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+'/transform', exist_ok=True)
    spacing=(image.dimlen['x'],image.dimlen['y'],image.dimlen['z'])
    
    parameterMapVector = sitk.VectorOfParameterMap()
    affine=sitk.GetDefaultParameterMap("affine")
    bspline=sitk.GetDefaultParameterMap("bspline")
    bspline['Metric0Weight']=(str(bweight),)
    bspline['FinalGridSpacingInPhysicalUnits']=(str(max(spacing)*bgrid),)
    #bspline["Metric"]=(*bspline["Metric"],"DisplacementMagnitudePenalty")
    if rms:
        bspline["Metric"]=("AdvancedMeanSquares",bspline["Metric"][1])
    #parameterMapVector.append(affine)
    parameterMapVector.append(bspline)
    
    
    '''start'''
    colorVec=False
    if 'RGB' in image.dim:
        colorVec=True

    fixImg=sitk.GetImageFromArray(np.copy(image.data[0]), isVector=colorVec)
    fixImg.SetOrigin(origin)
    fixImg.SetSpacing(spacing)
    sitk.WriteImage(fixImg,savePath+'/t'+str(refTimeInd)+'Img.mha')
    for n in range(refTimeInd):
        if n!=0:
            print('Registering t',n,' wrt t',n+1)
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOn()
            elastixImageFilter.LogToConsoleOff()
            fixImg=sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=colorVec)
            fixImg.SetOrigin(origin)
            fixImg.SetSpacing(spacing)
            movImg=sitk.GetImageFromArray(np.copy(image.data[n]), isVector=colorVec)
            movImg.SetOrigin(origin)
            movImg.SetSpacing(spacing)
            elastixImageFilter.SetFixedImage(fixImg)
            elastixImageFilter.SetMovingImage(movImg)
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            Tmap=elastixImageFilter.GetTransformParameterMap()
            for m in range(len(Tmap)):
                sitk.WriteParameterFile(Tmap[m],savePath+'/transform/t'+str(n+1)+'to'+str(n)+'_'+str(m)+'.txt')
        
        print('Registering t',0,' wrt t',n+1)
        elastixImageFilter=sitk.ElastixImageFilter()
        elastixImageFilter.LogToFileOn()
        elastixImageFilter.LogToConsoleOff()
        fixImg=sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=colorVec)
        fixImg.SetOrigin(origin)
        fixImg.SetSpacing(spacing)
        movImg=sitk.GetImageFromArray(np.copy(image.data[0]), isVector=colorVec)
        movImg.SetOrigin(origin)
        movImg.SetSpacing(spacing)
        elastixImageFilter.SetFixedImage(fixImg)
        elastixImageFilter.SetMovingImage(movImg)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.Execute()
        Tmap=elastixImageFilter.GetTransformParameterMap()
        for m in range(len(Tmap)):
            sitk.WriteParameterFile(Tmap[m],savePath+'/transform/t'+str(n+1)+'to0_'+str(m)+'.txt')

def inverseTransform(stlFile,refTimeInd,timeStepNo,mapNo,savePath='',TmapPath='',scale=1.,delimiter=' '):
    if savePath=='':
        savePath=stlFile[:-4]
    if TmapPath=='':
        TmapPath=savePath
    os.makedirs(savePath, exist_ok=True)
    if stlFile[-3:]=='stl':
        ref_mesh=trimesh.load(stlFile)
        oriPos=np.array(ref_mesh.vertices)/scale
    else:
        oriPos=np.loadtxt(stlFile,delimiter=delimiter)/scale
    np.savetxt(savePath+'/input0.pts',oriPos,header='point\n'+str(len(oriPos)),comments='')
    np.savetxt(savePath+'/input.pts',oriPos,header='point\n'+str(len(oriPos)),comments='')
    currentratio=1.
    calculatedPos=oriPos*0.
    for n in range(refTimeInd,0,-1):
        Tmap=[]#SimpleITK.VectorOfParameterMap()
        for m in range(mapNo):
            Tmap.append(sitk.ReadParameterFile(TmapPath+'/transform/t'+str(refTimeInd)+'to0_'+str(m)+'.txt'))
        
        transformixImageFilter=sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(Tmap)
        transformixImageFilter.SetMovingImage(sitk.ReadImage(TmapPath+'/t'+str(refTimeInd)+'Img.mha'))
        transformixImageFilter.SetFixedPointSetFileName(savePath+'/input0.pts')
        transformixImageFilter.SetOutputDirectory(savePath)
        transformixImageFilter.Execute()

        with open (savePath+'/outputpoints.txt', "r") as myfile:
            data=myfile.readlines()
        newPos=[]
        for string in data:
            result = re.search('OutputPoint(.*)Deformation', string)
            newPos.append(np.fromstring(result.group(1)[5:-6], sep=' '))
        newPos=np.array(newPos)
        ratio=timestepPoly(n,timeStepNo)
        calculatedPos+=newPos*currentratio*(1.-ratio)
        currentratio*=ratio
        if n>1 and n<(timeStepNo-1):
            Tmap=[]#SimpleITK.VectorOfParameterMap()
            for m in range(mapNo):
                Tmap.append(sitk.ReadParameterFile(TmapPath+'/transform/t'+str(n)+'to'+str(n-1)+'_'+str(m)+'.txt'))
            
            transformixImageFilter=sitk.TransformixImageFilter()
            transformixImageFilter.SetTransformParameterMap(Tmap)
            transformixImageFilter.SetMovingImage(sitk.ReadImage(TmapPath+'/t'+str(refTimeInd)+'Img.mha'))
            transformixImageFilter.SetFixedPointSetFileName(savePath+'/input.pts')
            transformixImageFilter.SetOutputDirectory(savePath)
            transformixImageFilter.Execute()

            with open (savePath+'/outputpoints.txt', "r") as myfile:
                data=myfile.readlines()
            newPos2=[]
            for string in data:
                result = re.search('OutputPoint(.*)Deformation', string)
                newPos2.append(np.fromstring(result.group(1)[5:-6], sep=' '))
            np.savetxt(savePath+'/input.pts',newPos2,header='point\n'+str(len(newPos2)),comments='')
    if stlFile[-3:]=='stl':
        ref_mesh.vertices=np.array(calculatedPos)*scale
        trimesh.io.export.export_mesh(ref_mesh,savePath+'/t0_reversedfrom_t'+str(refTimeInd)+'.stl')
    else:
        np.savetxt(savePath+'/t0_reversedfrom_t'+str(refTimeInd)+'.txt',np.array(calculatedPos)*scale)

def nlm_denoise(image,dim=['x','y','z'],refImg=None,spread=3,patch=1):
    image=image.clone()
    image.rearrangeDim(dim)
    if refImg!=type(None):
        refImg=refImg.clone()
        refImg.rearrangeDim(image.dim)
        if refImg.data.shape!=image.data.shape:
            raise ValueError('Reference image is not the same shape.')
        refImg_mean=np.mean(refImg.data,axis=tuple(range(len(dim))))
        image_mean=np.mean(refImg.data,axis=tuple(range(len(dim))))
        refImg_std=np.std(refImg.data,axis=tuple(range(len(dim))))
        image_std=np.std(refImg.data,axis=tuple(range(len(dim))))
        alpha=np.mean((refImg.data-refImg_mean)*(image.data-image_mean),axis=tuple(range(len(dim))))/refImg_std/image_std        
    else:
        alpha=np.zeros(image.data.shape[:len(dim)])
    newImg=image.clone()
    #################################################
    return newImg


class registrator:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self):
        self.fixImg=[]
        self.movImg=[]
        self.Tmap=[]
        self.savePath=[]
        self.similarity=[]
        self.Tmap_0=[]
    def createTransform(self,tPath='',regType='rigid',numOfReg=1):
        print(tPath)
        file1 = tPath + '/img2img_0.txt'
        if numOfReg==2:
            file2 = tPath + '/img2img_1.txt'
        
        fp = open(file1)
        with open(file1) as fp:
            for i, line in enumerate(fp):
                if i == 0:
                    p = line
                    tem = re.findall("-?\d*\.-?\d*",p)
                    cor1 = np.array(tem, dtype=np.float32)
                elif i == 22:
                    parameters = line
                    tform = re.findall("-?\d*\.-?\d*",parameters)
                    angles1 = np.array(tform[0:3], dtype=np.float64)
                    shift1 = np.array(tform[3:6], dtype=np.float64)
                    break
        fp.close()
        rigid_center1 = tuple([float("{0:.5f}".format(n)) for n in cor1])
        theta_x = angles1[0]
        theta_y = angles1[1]
        theta_z = angles1[2]
        rigid_translation1 = tuple([float("{0:.5f}".format(n)) for n in shift1])
        rigid_euler1 = sitk.Euler3DTransform(rigid_center1, theta_x, theta_y, theta_z, rigid_translation1)
        transform = rigid_euler1
        if numOfReg==2:
            
            fp = open(file2)
            if regType == 'rigid':
                with open(file2) as fp:
                    for i, line in enumerate(fp):
                        if i == 0:
                            p = line
                            tem = re.findall("-?\d*\.-?\d*",p)
                            cor2 = np.array(tem, dtype=np.float32)
                        elif i == 22:
                            parameters = line
                            tform = re.findall("-?\d*\.-?\d*",parameters)
                            angles2 = np.array(tform[0:3], dtype=np.float64)
                            shift2 = np.array(tform[3:6], dtype=np.float64)
                            break
                fp.close()    
                rigid_center2 = tuple([float("{0:.5f}".format(n)) for n in cor2])
                theta_x = angles2[0]
                theta_y = angles2[1]
                theta_z = angles2[2]
                rigid_translation2 = tuple([float("{0:.5f}".format(n)) for n in shift2])
                rigid_euler2 = sitk.Euler3DTransform(rigid_center2, theta_x, theta_y, theta_z, rigid_translation2)
                composite_transform = sitk.Transform(rigid_euler1)
                composite_transform.AddTransform(rigid_euler2)
            
            elif regType == 'affine':
                with open(file2) as fp:
                    for i, line in enumerate(fp):
                        if i == 0:
                            p = line
                            tem = re.findall("-?\d*\.-?\d*",p)
                            cor2 = np.array(tem, dtype=np.float32)
                        elif i == 21:
                            parameters = line
                            tform = re.findall("-?\d*\.-?\d*",parameters)
                            affine_matrix = np.array(tform[0:9], dtype=np.float64)
                            shift2 = np.array(tform[9:12], dtype=np.float64)
                            break
                        
                fp.close()    
                affine_center = tuple([float("{0:.5f}".format(n)) for n in cor2])
                affine_translation = tuple([float("{0:.5f}".format(n)) for n in shift2])
                affine = sitk.AffineTransform(affine_matrix, affine_translation, affine_center)        
                composite_transform = sitk.Transform(affine)
                composite_transform.AddTransform(rigid_euler1)
                transform = composite_transform
                self.Tmap_0=transform
        
        
    def register(self,image1,image2,initialTransf='',savePath='',regType='rigid',fileName='img2img',metric='rms',nres=6,smoothing=True,outputImage=True):
        
        image1=image1.clone()
        image2=image2.clone()
        image1.rearrangeDim(['z','y','x'])
        image2.rearrangeDim(['z','y','x'])
        origin1=(0.,0.,0.)
        origin2=(0.,0.,0.)
        
        if regType == 'affine':
            savePath = savePath + '_affine' 
        elif regType == 'rigid':
            savePath =  savePath + '_rigid'
            
        if metric == 'rms':
            savePath = savePath + '_rms' 
        elif metric == 'mi':
            savePath =  savePath + '_mi'
        elif metric == 'ncc':
            savePath =  savePath + '_ncc'
            
        savePath = savePath + '_' + str(nres) + 'res' 
        self.savePath = savePath
        
        if not(savePath):
            savePath=os.path.dirname(os.path.realpath(__file__))
           
        os.makedirs(savePath, exist_ok=True)
        os.makedirs(savePath+'/transform', exist_ok=True)
        spacing1=(image1.dimlen['x'],image1.dimlen['y'],image1.dimlen['z'])
        spacing2=(image2.dimlen['x'],image2.dimlen['y'],image2.dimlen['z'])
        
        parameterMapVector = sitk.VectorOfParameterMap()
        EulerTransform=sitk.GetDefaultParameterMap("rigid")
        EulerTransform["AutomaticScalesEstimation"]=( "true", ) 
        EulerTransform["AutomaticTransformInitialization"] = ( "true", ) 
        affine=sitk.GetDefaultParameterMap("affine")
        affine["AutomaticScalesEstimation"]=( "true", ) 
        affine["AutomaticTransformInitialization"] = ( "true", ) 
        
        if metric == 'rms':
            EulerTransform["Metric"]=("AdvancedMeanSquares",)
            affine["Metric"]=("AdvancedMeanSquares", )
        elif metric == 'mi':
            EulerTransform["Metric"]=("AdvancedMattesMutualInformation",)
            affine["Metric"]=("AdvancedMattesMutualInformation",)
        elif metric == 'ncc':
            EulerTransform["Metric"]=("AdvancedNormalizedCorrelation",)
            affine["Metric"]=("AdvancedNormalizedCorrelation",)
        
        EulerTransform["Registration"] = ("MultiResolutionRegistration",)
        EulerTransform["FixedImagePyramid"] =("FixedSmoothingImagePyramid",)
        EulerTransform["MovingImagePyramid"] =("MovingSmoothingImagePyramid",)
        affine["Registration"] = ("MultiResolutionRegistration",)
        affine["FixedImagePyramid"] =("FixedSmoothingImagePyramid",)
        affine["MovingImagePyramid"] =("MovingSmoothingImagePyramid",)
        
        EulerTransform['MaximumNumberOfIterations'] = ['1024']
        affine['MaximumNumberOfIterations'] = ['1024']
        if nres == 4:
            EulerTransform["NumberOfResolutions"] = ("4",)
            affine["NumberOfResolutions"] = ("4",)
            if smoothing:
                EulerTransform["FixedImagePyramidSchedule"] = ('8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
                EulerTransform["MovingImagePyramidSchedule"] = ('8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
                affine["FixedImagePyramidSchedule"] = ('8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
                affine["MovingImagePyramidSchedule"] = ('8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
        elif nres == 5:
            EulerTransform["NumberOfResolutions"] = ("5",)
            affine["NumberOfResolutions"] = ("5",)
            if smoothing:
                EulerTransform["FixedImagePyramidSchedule"] = ('16' '16' '16' '8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
                EulerTransform["MovingImagePyramidSchedule"] = ('16' '16' '16' '8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
                affine["FixedImagePyramidSchedule"] = ('16' '16' '16' '8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
                affine["MovingImagePyramidSchedule"] = ('16' '16' '16' '8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
        elif nres == 6:
            EulerTransform["NumberOfResolutions"] = ("6",)
            affine["NumberOfResolutions"] = ("6",)
            if smoothing:
                EulerTransform["FixedImagePyramidSchedule"] = ('32' '32' '32' '16' '16' '16' '8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
                EulerTransform["MovingImagePyramidSchedule"] = ('32' '32' '32' '16' '16' '16' '8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
                affine["FixedImagePyramidSchedule"] = ('32' '32' '32' '16' '16' '16' '8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
                affine["MovingImagePyramidSchedule"] = ('32' '32' '32' '16' '16' '16' '8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
        elif nres == 7:
            EulerTransform["NumberOfResolutions"] = ("7",)
            affine["NumberOfResolutions"] = ("7",)
            if smoothing:
                EulerTransform["FixedImagePyramidSchedule"] = ('64' '64' '64' '32' '32' '32' '16' '16' '16' '8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
                EulerTransform["MovingImagePyramidSchedule"] = ('64' '64' '64' '32' '32' '32' '16' '16' '16' '8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
                affine["FixedImagePyramidSchedule"] = ('64' '64' '64' '32' '32' '32''16' '16' '16' '8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
                affine["MovingImagePyramidSchedule"] = ('64' '64' '64' '32' '32' '32''16' '16' '16' '8' '8' '8' '4' '4' '4' '2' '2' '2' '1' '1' '1')
    
        parameterMapVector.append(EulerTransform)
        if regType == 'affine':
            parameterMapVector.append(affine)
        #elif regType == 'rigid':
            #parameterMapVector.append(EulerTransform)
        
        
        colorVec=False
        if 'RGB' in image1.dim:
            colorVec=True
        
        elastixImageFilter=sitk.ElastixImageFilter()
        elastixImageFilter.LogToFileOff()
        elastixImageFilter.LogToConsoleOff()
        
        x = np.copy(image1.data)
        fixImg=sitk.GetImageFromArray(x.astype(np.uint8), isVector=colorVec)
        
        fixImg.SetOrigin(origin1)
        fixImg.SetSpacing(spacing1)
        if outputImage:
            sitk.WriteImage(fixImg,savePath+'/'+fileName+'_fixImg.mha')
        
        movImg=sitk.GetImageFromArray(np.copy(image2.data.astype(np.uint8)), isVector=colorVec)
        movImg.SetOrigin(origin2)
        movImg.SetSpacing(spacing2)
        
        
        elastixImageFilter.SetFixedImage(fixImg)
        
        
        if (initialTransf):
            if type(initialTransf)==str:
                elastixImageFilter.SetInitialTransformParameterFileName(initialTransf)
                '''
                #tPath = initialTransf
                #file1 = tPath + '/img2img_0.txt'
                file1 = initialTransf
                fp = open(file1)
                with open(file1) as fp:
                    for i, line in enumerate(fp):
                        if i == 0:
                            p = line
                            tem = re.findall("-?\d*\.-?\d*",p)
                            cor1 = np.array(tem, dtype=np.float32)
                        elif i == 22:
                            parameters = line
                            tform = re.findall("-?\d*\.-?\d*",parameters)
                            angles1 = np.array(tform[0:3], dtype=np.float64)
                            shift1 = np.array(tform[3:6], dtype=np.float64)
                            break
                fp.close()
                rigid_center1 = tuple([float("{0:.5f}".format(n)) for n in cor1])
                theta_x = angles1[0]
                theta_y = angles1[1]
                theta_z = angles1[2]
                rigid_translation1 = tuple([float("{0:.5f}".format(n)) for n in shift1])
                rigid_euler1 = sitk.Euler3DTransform(rigid_center1, theta_x, theta_y, theta_z, rigid_translation1)
                transform = rigid_euler1
                interpolator = sitk.sitkLanczosWindowedSinc
                outimage = sitk.Resample(movImg, fixImg, transform, interpolator, 0)
                movImg = outimage
                '''
            elif type(initialTransf) in [float,int]:#rotate y in deg
                rigid_center1 = tuple([image2.dimlen['x']*image2.data.shape[2]/2.,image2.dimlen['y']*image2.data.shape[1]/2.,image2.dimlen['z']*image2.data.shape[0]/2.])
                theta_x = 0.0
                theta_y = initialTransf*np.pi/180.
                theta_z = 0.0
                rigid_translation1 = tuple([0.0,0.0,0.0])
                rigid_euler1 = sitk.Euler3DTransform(rigid_center1, theta_x, theta_y, theta_z, rigid_translation1)
                transform = rigid_euler1
                interpolator = sitk.sitkLanczosWindowedSinc
                outimage = sitk.Resample(movImg, fixImg, transform, interpolator, 0)
                movImg = outimage
        if outputImage:
            sitk.WriteImage(movImg,savePath+'/'+fileName+'_movImg.mha')
        elastixImageFilter.SetMovingImage(movImg)
        self.fixImg = fixImg
        self.movImg = movImg
        
        elastixImageFilter.SetParameterMap(parameterMapVector)
        
        os.makedirs(savePath+'/'+'logFile', exist_ok=True)
        
        #elastixImageFilter.SetLogToFile(True)
        #elastixImageFilter.SetOutputDirectory(savePath+'/'+'logFile')
        #elastixImageFilter.SetLogFileName('elastix.txt')
        elastixImageFilter.SetLogToConsole(False)

        elastixImageFilter.Execute()
        Tmap=elastixImageFilter.GetTransformParameterMap()
        self.Tmap = Tmap
        
            
        for m in range(len(Tmap)):
            sitk.WriteParameterFile(Tmap[m],os.path.join(savePath,'transform',fileName+'_'+str(m)+'.txt'))
            sitk.PrintParameterMap(Tmap[m])
            elastixImageFilter.PrintParameterMap(Tmap[m])
                
        resImg=elastixImageFilter.GetResultImage()
        spacingR = resImg.GetSpacing()
        originR = resImg.GetOrigin()
        arr=sitk.GetArrayFromImage(resImg)
        resImg=sitk.GetImageFromArray(np.copy(arr.astype(np.uint8)), isVector=colorVec)
        resImg.SetOrigin(originR)
        resImg.SetSpacing(spacingR)
        if outputImage:
            sitk.WriteImage(resImg,savePath+'/'+fileName+'_resultImg.mha')
        
        similarity_filter = sitk.SimilarityIndexImageFilter()
        similarity_filter.Execute(fixImg, resImg)
        res = similarity_filter.GetSimilarityIndex()
        
        fo = savePath+'/similarity.txt'
        myFile = open(fo, 'w')  
        myFile.write('Similarity index: '+str(res)) 
        myFile.close()
        self.similarity = res   
        
    def transform(self,image_f=None,image_m=None,TmapPath='',trPath='',regType=None,interpolator=None,expandFactor=1):
        if type(image_f) == type(None):
            image_f = self.fixImg
        else:
            image_f.rearrangeDim(['z','y','x'])
            origin = (0.,0.,0.)
            spacingf = (image_f.dimlen['x'],image_f.dimlen['y'],image_f.dimlen['z'])
            a = np.copy(image_f.data)
            imgf = sitk.GetImageFromArray(a.astype(np.float32), isVector=False)
            imgf.SetOrigin(origin)
            imgf.SetSpacing(spacingf)
            image_f = imgf
        if type(image_m) == type(None):
            image_m = self.movImg
        else:
            image_m.rearrangeDim(['z','y','x'])
            origin = (0.,0.,0.)
            spacingm = (image_m.dimlen['x'],image_m.dimlen['y'],image_m.dimlen['z'])
            b = np.copy(image_m.data)
            imgm = sitk.GetImageFromArray(b.astype(np.float32), isVector=False)
            imgm.SetOrigin(origin)
            imgm.SetSpacing(spacingm)
            image_m = imgm
            
        if type(regType) == type(None):
            regType = 'rigid'
            
        if not trPath:
            trPath = os.path.join(self.savePath,'transform')
            
        path, dirs, files = next(os.walk(trPath))
        numOfReg = len(files)
            
        if TmapPath == '':
            file1 = os.path.join(trPath,files[0])
            if numOfReg==2:
                file2 = os.path.join(trPath,files[1])
            fp = open(file1)
            with open(file1) as fp:
                for i, line in enumerate(fp):
                    if i == 0:
                        p = line
                        tem = re.findall("-?\d*\.-?\d*",p)
                        cor1 = np.array(tem, dtype=np.float32)
                    elif i == 22:
                        parameters = line
                        tform = re.findall("-?\d*\.-?\d*",parameters)
                        angles1 = np.array(tform[0:3], dtype=np.float64)
                        shift1 = np.array(tform[3:6], dtype=np.float64)
                        break
            fp.close()
            rigid_center1 = tuple([float("{0:.5f}".format(n)) for n in cor1])
            theta_x = angles1[0]
            theta_y = angles1[1]
            theta_z = angles1[2]
            rigid_translation1 = tuple([float("{0:.5f}".format(n)) for n in shift1])
            rigid_euler1 = sitk.Euler3DTransform(rigid_center1, theta_x, theta_y, theta_z, rigid_translation1)
            transform = rigid_euler1
            if numOfReg==2:
                fp = open(file2)
                if regType == 'rigid':
                    with open(file2) as fp:
                        for i, line in enumerate(fp):
                            if i == 0:
                                p = line
                                tem = re.findall("-?\d*\.-?\d*",p)
                                cor2 = np.array(tem, dtype=np.float32)
                            elif i == 22:
                                parameters = line
                                tform = re.findall("-?\d*\.-?\d*",parameters)
                                angles2 = np.array(tform[0:3], dtype=np.float64)
                                shift2 = np.array(tform[3:6], dtype=np.float64)
                                break
                    fp.close()    
                    rigid_center2 = tuple([float("{0:.5f}".format(n)) for n in cor2])
                    theta_x = angles2[0]
                    theta_y = angles2[1]
                    theta_z = angles2[2]
                    rigid_translation2 = tuple([float("{0:.5f}".format(n)) for n in shift2])
                    rigid_euler2 = sitk.Euler3DTransform(rigid_center2, theta_x, theta_y, theta_z, rigid_translation2)
                    composite_transform = sitk.Transform(rigid_euler2)
                    composite_transform.AddTransform(rigid_euler1)
                    transform = composite_transform
                
                elif regType == 'affine':
                    with open(file2) as fp:
                        for i, line in enumerate(fp):
                            if i == 0:
                                p = line
                                tem = re.findall("-?\d*\.-?\d*",p)
                                cor2 = np.array(tem, dtype=np.float32)
                            elif i == 21:
                                parameters = line
                                tform = re.findall("-?\d*\.-?\d*",parameters)
                                affine_matrix = np.array(tform[0:9], dtype=np.float64)
                                shift2 = np.array(tform[9:12], dtype=np.float64)
                                break
                            
                    fp.close()    
                    affine_center = tuple([float("{0:.5f}".format(n)) for n in cor2])
                    affine_translation = tuple([float("{0:.5f}".format(n)) for n in shift2])
                    affine = sitk.AffineTransform(affine_matrix, affine_translation, affine_center)        
                    composite_transform = sitk.Transform(affine)
                    composite_transform.AddTransform(rigid_euler1)
                    transform = composite_transform
            
                
        dx = min(list(image_f.GetSpacing()))
        dz = max(list(image_f.GetSpacing()))
        output_spacing = [dx, dx, dx]
        output_size = list(image_f.GetSize())
        output_size[2] = output_size[2]*round(dz/dx)
        output_size = [int(i * expandFactor) for i in output_size]
        print(output_size)
        size = max(output_size)
        output_size = tuple([size,size,size])
        output_origin = [0, 0, 0]
        output_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        if type(interpolator) == type(None):
            interpolator = sitk.sitkLanczosWindowedSinc
        
        outimage = sitk.Resample(image_m, output_size, transform, interpolator, output_origin, output_spacing,output_direction)
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator = interpolator
        resample.SetOutputDirection = output_direction
        resample.SetOutputOrigin = output_origin
        resample.SetOutputSpacing(output_spacing)
        resample.SetSize(output_size)
        newimage = resample.Execute(image_f)
        
        sitk.WriteImage(newimage,os.path.join(self.savePath,'fixed_resampled.mha'))
        sitk.WriteImage(outimage,os.path.join(self.savePath,'moving_transformed_resampled.mha'))

        
class SAC:
    def __init__(self,cI,returnStats=False):
        if isinstance(cI,list):
            self.cI=cI[0]
            self.func=cI[1]
        else:
            self.cI=cI
            self.func=np.mean
        if self.cI>1:
            raise Exception('Confidence Interval > 1!')
        self.returnStats=returnStats
    def __call__(self,val):
        val=np.array(val)
        val=val[val>=1]
        if len(val)==0:
            if self.returnStats:
                value=np.array([0,0])
            else:
                value=0
            #print('no value at coords',x,y,z)
        elif (val.max()-val.min())<=3:
            if self.returnStats:
                value=np.array([self.func(val),0])
            else:
                value=self.func(val)
        else:
            value=self.func(val)
            bincount= np.bincount(np.around(val).astype(int))
            bincountCompressed = bincount[bincount!=0]
            intensityCompresed=np.nonzero(bincount)[0]
            CumSumCompresed=np.insert(np.cumsum(bincountCompressed), 0, 0)
            lowerBoundInd=1
            upperBoundInd=len(bincountCompressed)
            totalCount=bincountCompressed.sum()
            '''#old
            while bincountCompressed[:lowerBoundInd].sum()<(totalCount*self.cI):
                lowerBoundInd+=1
            while bincountCompressed[(upperBoundInd-1):].sum()<(totalCount*self.cI):
                upperBoundInd-=1
            bound=np.zeros((257,4),dtype=int)
            for low in range(upperBoundInd):
                for high in range(max(low+1,lowerBoundInd),len(bincountCompressed)+1):
                    width=intensityCompresed[high-1]-intensityCompresed[low]+1
                    temp_sum=bincountCompressed[low:high].sum()
                    if temp_sum>=(totalCount*self.cI):
                        if temp_sum>np.abs(bound[width,0]):
                            bound[width]=np.array([temp_sum,low,high,width])
                        elif temp_sum==np.abs(bound[width,0]):
                            bound[width]=np.array([-temp_sum,1,0,width])
            '''
            while CumSumCompresed[lowerBoundInd]<(totalCount*self.cI):
                lowerBoundInd+=1
            while (CumSumCompresed[-1]-CumSumCompresed[upperBoundInd-1])<(totalCount*self.cI):
                upperBoundInd-=1
            bound=np.zeros((257,4),dtype=int)
            for low in range(upperBoundInd):
                for high in range(max(low+1,lowerBoundInd),len(bincountCompressed)+1):
                    width=intensityCompresed[high-1]-intensityCompresed[low]+1
                    temp_sum=CumSumCompresed[high]-CumSumCompresed[low]
                    if temp_sum>=(totalCount*self.cI):
                        if temp_sum>np.abs(bound[width,0]):
                            bound[width]=np.array([temp_sum,low,high,width])
                        elif temp_sum==np.abs(bound[width,0]):
                            bound[width]=np.array([-temp_sum,1,0,width])
            newbound=bound[bound[:,0]>0]
            if len(newbound)>0:
                for n in range(len(newbound)):
                    if newbound[n,0]>np.abs(bound[:newbound[n,3],0]).max():
                        high=newbound[n,2]
                        low=newbound[n,1]
                        inliersInd=np.logical_and(np.array(val)<=intensityCompresed[high-1],np.array(val)>=intensityCompresed[low])
                        value=self.func(val[inliersInd])
                        if self.returnStats:
                            temp_bincount=bincount.copy()
                            temp_bincount[intensityCompresed[low]:(intensityCompresed[high-1]+1)]=0
                            cs=np.insert(np.cumsum(temp_bincount), 0, 0)
                            cs2=np.roll(cs,-newbound[n,3])
                            value=np.array([value,np.max(cs2-cs)])
                        break
            elif self.returnStats:
                value=np.array([value,0])
                
        return value
def getPhaseCong(image,saveTo,alpha=0.6,minWaveLength=5,schemeArgs=None):
    print('Please rearrange the axes in this order [two other axes... , main direction of beam,2D perpendicular direction]')
    print('Current axis arrangement:',image.dim)
    #Scheme arguements = None => beam parallel to y aiming y+
    #Scheme arguements = float => beam at an angle towards y+ with x+ positive
    #Scheme arguements = [y,x] => beam originate at y,x

    import phasepack
    phaseC_PC=np.zeros(image.data.shape)
    phaseC_ori=np.zeros(image.data.shape)
    for nt in range(image.data.shape[0]):
        for nz in range(image.data.shape[1]):
            phaseC_PC[nt,nz],phaseC_ori[nt,nz],ft_temp,T_temp=phasepack.phasecongmono(image.data[nt,nz],minWaveLength=minWaveLength)
    phaseC_ori=(phaseC_ori/90.-1.)*np.pi/2. #0 aligns to y axis ,clockwise positive
    
    M=image.clone()
    M.data=phaseC_ori
    S=image.clone()
    S.data=phaseC_PC
    if type(schemeArgs)!=type(None):
        if len(schemeArgs)==1:
            temp_angle=np.ones((image.data.shape[-2:]))*schemeArgs[0]
        else:
            temp_angle=np.arctan2(np.tile((np.arange(image.data.shape[-1])*image.dimlen[image.dim[-1]]-schemeArgs[1]),(image.data.shape[-2],1)),np.tile((np.arange(image.data.shape[-2])*image.dimlen[image.dim[-2]]-schemeArgs[0]).reshape((-1,1)),(1,image.data.shape[-1])))
        temp_angle[temp_angle>(np.pi/2.)]-=np.pi
        temp_angle[temp_angle<(-np.pi/2.)]+=np.pi
        M.data=-M.data+temp_angle
        M.data[M.data>(np.pi/2.)]-=np.pi
        M.data[M.data<(-np.pi/2.)]+=np.pi
    M.data=np.abs(M.data)/(np.pi/2.)
    
    os.makedirs(saveTo, exist_ok=True)
    M.save(saveTo+'/phaseC_ori')
    S.save(saveTo+'/phaseC_PC')
    
def compound(image,scheme='mean',schemeArgs=None,axis='t',twoD=False,parallel=True,returnStats=False):
    image=image.clone()
    image.rearrangeDim([axis],arrangeFront=False)
    resultImage=image.clone()
    resultImage.removeDim(axis)
    if scheme=='mean':
        if type(schemeArgs)==type(None):
            resultImage.data=image.data.mean(axis=-1)
        else:
            schemeArgs=schemeArgs/np.sum(schemeArgs)
            resultImage.data=np.sum(image.data*schemeArgs,axis=-1)
    elif scheme=='SAC':
        if type(schemeArgs)==type(None):
            schemeArgs=0.5
            
        SCAfunc=SAC(schemeArgs,returnStats=returnStats)
        
        resultData=image.data.reshape((-1,image.data.shape[-1]),order='F')
        if twoD:
            for xn in range(image.data.shape[0]):
                print('    {0:.3f}% completed...'.format(float(xn)/image.data.shape[0]*100.))
                for yn in range(image.data.shape[1]):
                    resultImage.data[xn,yn]=SCAfunc(image.data[xn,yn])
                    #for zn in range(image.data.shape[2]):
                    #    resultImage.data[xn,yn,zn]=SCAfunc(image.data[xn,yn,zn])
        elif not(parallel):
            for xn in range(image.data.shape[0]):
                print('    {0:.3f}% completed...'.format(float(xn)/image.data.shape[0]*100.))
                for yn in range(image.data.shape[1]):
                    for zn in range(image.data.shape[2]):
                        resultImage.data[xn,yn,zn]=SCAfunc(image.data[xn,yn,zn])
        else:
        
            pool = multiprocessing.Pool()
            resultdata=np.array(pool.map(SCAfunc,resultData))
            pool.close()
            pool.join()
            resultImage.data=resultdata.reshape((*image.data.shape[:-1],*resultdata.shape[1:]),order='F')
        
    elif scheme=='phase':
        #Scheme arguements = [alpha,S,M] alpha=0.6
        schemeArgs[1].rearrangeDim([axis],arrangeFront=False)
        schemeArgs[2].rearrangeDim([axis],arrangeFront=False)
        if image.data.shape[-1]<3:
            S1=schemeArgs[1].data
            S2=np.roll(S1,1,axis=-1)
            S1r=1.-S1
            S2r=1.-S2
            M1=schemeArgs[2].data
            M2=np.roll(M1,1,axis=-1)
            M1r=1.-M1
            M2r=1.-M2
            weight=S1*(S2r+S2*(M1*(M2r+0.5*M2)+0.5*M1r*M2r))+schemeArgs[0]/2.*S1r*S2r
            resultImage.data=np.average(image.data,axis=-1,weights=weight)
        else:
            maxInd=np.argsort(image.data,axis=-1)[...,-3:].reshape(-1)
            S1=schemeArgs[1].data.reshape((-1,schemeArgs[1].data.shape[-1]))[np.repeat(range(np.prod(schemeArgs[1].data.shape[:-1])),3),maxInd].reshape((*schemeArgs[1].data.shape[:-1],3))
            M1=schemeArgs[2].data.reshape((-1,schemeArgs[2].data.shape[-1]))[np.repeat(range(np.prod(schemeArgs[2].data.shape[:-1])),3),maxInd].reshape((*schemeArgs[2].data.shape[:-1],3))
            S2=np.roll(S1,1,axis=-1)
            S3=np.roll(S1,2,axis=-1)
            S1r=1.-S1
            S2r=1.-S2
            S3r=1.-S3
            M2=np.roll(M1,1,axis=-1)
            M3=np.roll(M1,2,axis=-1)
            M1r=1.-M1
            M2r=1.-M2
            M3r=1.-M3
            weight=S1*(S2r*S3r+S2*S3r*M1*(M2r+0.5*M2)+S2r*S3*M1*(M3r+0.5*M3)+S2*S3*(M1*(M2r*M3r+0.5*(M2*M3r+M2r*M3)+1./3.*M2*M3)+M1r*M2r*M3r))+schemeArgs[0]/3.*S1r*S2r*S3r
            resultImage.data=np.average(image.data.reshape((-1,image.data.shape[-1]))[np.repeat(range(np.prod(image.data.shape[:-1])),3),maxInd].reshape((*image.data.shape[:-1],3)),axis=-1,weights=weight)
        
        
    elif scheme=='median':
        resultImage.data=np.median(image.data,axis=-1)
    elif scheme=='wavelet':
        coef=[]
        for n in range(image.data.shape[-1]):
            coef.append(pywt.dwtn(image.data[...,n],'haar'))
        resultCoef={}
        for key in coef[0].keys():
            if 'd' not in key:
                resultCoef[key]=coef[0][key]
                for n in range(1,image.data.shape[-1]):
                    resultCoef[key]=np.maximum(resultCoef[key],coef[n][key])
            else:
                resultCoef[key]=coef[n][key]/float(image.data.shape[-1])
                for n in range(1,image.data.shape[-1]):
                    resultCoef[key]+=coef[n][key]/float(image.data.shape[-1])
        resultImage.data=pywt.idwtn(resultCoef,'haar')
    elif scheme=='maximum':
        resultImage.data=image.data.max(axis=-1)
    elif scheme=='minimum':
        resultImage.data=image.data.min(axis=-1)
    else:
        raise Exception('No Valid Scheme Chosen.')
            
    return resultImage
