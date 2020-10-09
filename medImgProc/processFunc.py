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
    Author: w.x.chan@gmail.com         30OCT2019           - v1.6.6
                                                              -corrected masking of alignaxis_translate
                                                              -corrected typo of "alignaxis_translate" in 2 location
    Author: w.x.chan@gmail.com         30OCT2019           - v1.7.0
                                                              -added class alignAxesClass to be used for KalmanFilter (for module "motionSegmentation")
                                                              -return new position for transform_img2img
    Author: w.x.chan@gmail.com         31OCT2019           - v1.7.3
                                                              -in transform_img2img, assign moving img when not specified
    Author: w.x.chan@gmail.com         10NOV2019           - v1.8.3
                                                              -in alignAxes, included calFill where calFill=None, pixel outside border of translated image is filled with mean intensity 
                                                              -in alignAxes, for Eulerian scheme and includeRotate = False, accumulate translation to prevent data loss
    Author: w.x.chan@gmail.com         13NOV2019           - v1.8.5
                                                              -in alignAxes, rearrange mask
    Author: w.x.chan@gmail.com         13NOV2019           - v1.9.0
                                                              -in TmapRegister, added mode to do cyclic
    Author: w.x.chan@gmail.com         18NOV2019           - v1.9.1
                                                              -in alignAxes, added option to initTranslate
    Author: w.x.chan@gmail.com         18NOV2019           - v2.0.0
                                                              -in alignAxes, added multilevel nres
    Author: w.x.chan@gmail.com         18NOV2019           - v2.1.1
                                                              -in alignAxes, debug multilevel nres which gave different shape
                                                              -in registrator, added twoD ndarray input
    Author: w.x.chan@gmail.com         18NOV2019           - v2.1.3
                                                              -in gradient descent, debug maxTranslate
                                                              -in gradient descent, added cap on gradient to 10* errThreshold
                                                              -in alignAxes, debug multilevel nres of mask
    Author: w.x.chan@gmail.com         19NOV2019           - v2.1.4
                                                              -in alignAxes, debug boolean of translateaxes
    Author: w.x.chan@gmail.com         19NOV2019           - v2.1.5
                                                              -change print to logging
    Author: w.x.chan@gmail.com         19NOV2019           - v2.1.7
                                                              -in alignAxes, debug boolean of translateaxes
    Author: w.x.chan@gmail.com         19NOV2019           - v2.2.3
                                                              -in transform_img2img, allow image np.ndarray input
    Author: w.x.chan@gmail.com         11DEC2019           - v2.2.4
                                                              -in transform_img2img, edit not to savetxt of oripos
    Author: w.x.chan@gmail.com         02JAN2020           - v2.2.8
                                                              -in alignAxes_translate, add mask type tuple
    Author: w.x.chan@gmail.com         02JAN2020           - v2.3.14
                                                              -in translateArray, include interpolation order arguments
    Author: w.x.chan@gmail.com         04FEB2020           - v2.4.0
                                                              -added functions: cyclicNonRigidCorrection AND nonRigidRegistration
    Author: w.x.chan@gmail.com         17FEB2020           - v2.4.5
                                                              -in transform_img2img, debug image np.ndarray input without scaling                                                            
    Author: w.x.chan@gmail.com         26FEB2020           - v2.4.6
                                                              -in TmapRegister, save fileScale                                                           
    Author: w.x.chan@gmail.com         28FEB2020           - v2.5.0
                                                              -in functions with SimpleITK image registration, add maskArray input
    Author: w.x.chan@gmail.com         05MAR2020           - v2.5.1
                                                              -in SAC, sanitise data to 'uint8'
    Author: w.x.chan@gmail.com         05MAR2020           - v2.5.6
                                                              -in transform, added forwardbackward and change default startTime from 1 to 0,
    Author: w.x.chan@gmail.com         19MAR2020           - v2.5.7
                                                              -in compound, bound image data before change type to 'uint8'
    Author: w.x.chan@gmail.com         25MAR2020           - v2.6.16
                                                              -in compound, bound 
    Author: w.x.chan@gmail.com         31MAR2020           - v2.6.18
                                                              -in compound, filter out float('nan'),added None for SAC.func to return inliers with outliers being nan
    Author: w.x.chan@gmail.com         21JUL2020           - v2.6.23
                                                              -in registrator: register, added maskArray1 and maskArray2
    Author: w.x.chan@gmail.com         22JUL2020           - v2.6.24
                                                              -in registrator: register, change sampler to RandomSparseMask is mask exist
    Author: w.x.chan@gmail.com         09OCT2020           - v2.6.26
                                                              -(debug) gradient ascent add self.errThreshold
    Author: w.x.chan@gmail.com         09OCT2020           - v2.6.27
                                                              -gradient ascent add f_val error
    Author: w.x.chan@gmail.com         09OCT2020           - v2.6.28
                                                              -gradient ascent, debug finetune_space=0 error
    Author: w.x.chan@gmail.com         09OCT2020           - v2.6.29
                                                              -gradient descent, debug f_error
                                                              -gradient ascent added warning when maximum iteration reached
    Author: w.x.chan@gmail.com         09OCT2020           - v2.6.30
                                                              -gradient descent, debug finetune_space
    Author: w.x.chan@gmail.com         09OCT2020           - v2.6.32
                                                              -gradient descent, add normalize
                                                              
Requirements:
    numpy.py
    scipy.py
    cv2.py (opencv)

Known Bug:
    color format not supported
    last point of first axis ('t') not recorded in snapDraw_black
All rights reserved.
'''
_version='2.6.32'

import logging
logger = logging.getLogger(__name__)
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
    def __init__(self,func,initPara,args=(),gain=None,errThreshold=0.6,f_error=float('inf'),limitRun=100,maxPara=None,minPara=None,finetune_space=2,normalize_para=False):
        self.func=func
        self.para=np.array(initPara)
        self.args=args
        self.paraLength=len(initPara)
        if type(errThreshold) in [int,float]:
            self.errThreshold=np.ones(len(initPara))*errThreshold
        else:
            self.errThreshold=errThreshold
        self.limitRun=limitRun
        self.slope=1.
        self.finetune_space=finetune_space
        self.fVal=0.
        self.f_error=f_error
        if maxPara is None:
            maxPara=np.array([float('inf')]*self.paraLength)
        if minPara is None:
            minPara=np.array([float('-inf')]*self.paraLength)
        self.maxPara=maxPara
        self.minPara=minPara
        self.normalize_para=normalize_para
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
            logger.info('Initial value= '+str(self.fVal)+', with '+str(self.para))
        for count in range(1,self.limitRun):
            if error<1.:
                break
            #fValtemp=func(self.para,*self.args)
            gradient=self.grad()
            if self.normalize_para:
                '''normalize gradient with errThreshold'''
                gradient=gradient/self.errThreshold
            else:
                '''reduce gradient (smoothing)'''
                for n in range(len(gradient)):
                    if (self.gain*self.slope*gradient[n]) > (self.errThreshold[n]*10.):
                        gradient[n]=self.errThreshold[n]*10./self.gain/self.slope
                    elif (self.gain*self.slope*gradient[n]) < (-self.errThreshold[n]*10.):
                        gradient[n]=-self.errThreshold[n]*10./self.gain/self.slope
            newPara=self.para+self.gain*self.slope*gradient
            '''reduce gain for max and minPara'''
            for n in range(len(self.para)):
                if newPara[n] > self.maxPara[n]:
                    self.gain=min(self.gain,np.abs((self.maxPara[n]-self.para[n])/self.slope/gradient[n]))
                elif newPara[n] < self.minPara[n]:
                    self.gain=min(self.gain,np.abs((self.minPara[n]-self.para[n])/self.slope/gradient[n]))
            
            newPara=self.para+self.gain*self.slope*gradient
            newfVal=self.func(newPara,*self.args)
            logger.debug(str(count)+' : Current Para '+str(self.para)+' , Cost '+str(self.fVal)+' , Gradient '+str(gradient)+' , Gain '+str(self.gain)+' , Next cost '+str(newfVal))
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
            if ((newfVal<self.fVal) ^ (self.slope==1)):
                self.gain*=1.3
            '''calculate error and update'''
            error=max(np.max(np.abs(newPara-self.para)/self.errThreshold),abs(self.fVal-newfVal)*2./max(abs(newfVal),abs(self.fVal))/self.f_error)
            self.para=np.copy(newPara)
            self.fVal=newfVal
            if count%report==0:
                logger.info('iteration '+str(count)+', value= '+str(self.fVal))
        else:
            logger.warning('Maximum iteration ('+str(self.limitRun)+') reached.')
        '''fine tune adjustment by errThreshold'''
        gradient=self.grad()
        if self.finetune_space>0:
            logger.info('fine tuning')
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
                    gradient=self.grad()
                else:
                    break
        if report<float('inf'):
            logger.info('Final value= '+str(self.fVal)+', with '+str(self.para))
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
    def __init__(self,func,initPara,args=(),gain=None,errThreshold=1.,f_error=float('inf'),limitRun=100,maxPara=None,minPara=None,finetune_space=2,normalize_para=False):
        super(gradient_descent, self).__init__(func,initPara,args=args,gain=gain,errThreshold=errThreshold,f_error=f_error,limitRun=limitRun,maxPara=maxPara,minPara=minPara,finetune_space=finetune_space,normalize_para=normalize_para)
        self.slope=-1.
'''
internal functions
'''
def calculate_correlation(imageAArray,imageBArray,maskArray):
    '''Calculate the correlation of two images with variance of intensity'''
    imageAArray=np.copy(imageAArray)
    imageBArray=np.copy(imageBArray)
    total=maskArray.sum()
    meanA=np.sum(imageAArray*maskArray)/total
    meanB=np.sum(imageBArray*maskArray)/total
    stdA=np.sqrt(np.sum((imageAArray-meanA)**2*maskArray)/total)
    stdB=np.sqrt(np.sum((imageBArray-meanB)**2*maskArray)/total)
    
    correl_val=np.sum((imageAArray-meanA)*(imageBArray-meanB)*maskArray)/stdA/stdB
    return correl_val
    
def translateArray(oldArray,translateLastaxes,includeRotate,fill,order=3):
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
    newArray=shift(oldArray,sanitized_translateIndex,cval=fill,order=order)
    axis=[len(oldArray.shape)-1,len(oldArray.shape)-2]
    for n in range(trlen,len(translateLastaxes)):
        newArray=rotate(newArray,translateLastaxes[n],axes=axis,cval=fill,reshape=False,order=order)
        if (axis[0]-1)<=axis[1]:
            axis[1]-=1
            axis[0]=len(oldArray.shape)-1
        else:
            axis[0]-=1
    return newArray
def correlationFunc_translate(translateLastaxes,arrayA,arrayB,mask,includeRotate=False,calFill=0):
    '''
    mask=[(min1stdim,max1stdim),(min2nddim,max2nddim)...]
    '''
    newArrayB=translateArray(arrayB,translateLastaxes,includeRotate,calFill)
    corr=calculate_correlation(arrayA,newArrayB,mask)
    return corr
    
def correlation_translate(arrayA,arrayB,translateLimit,initialTranslate=None,includeRotate=False,calFill=0,mask=None):
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
    if type(mask)!=type(None):
        if type(mask)!=np.ndarray:
            if type(mask[0])!=slice:
                sliceList=[]
                for n in range(len(mask)):
                    sliceList.append(slice(mask[n][0],mask[n][1]))
            else:
                sliceList=mask
            mask=np.ones(arrayA.shape)
            mask[sliceList]=0
        elif mask.shape!=arrayA.shape:
            maskArray=np.zeros(arrayA.shape)
            maskArray[mask]=1
            mask=maskArray
            #maskVal=np.sum((imageAArray[sliceList]-meanA)*(imageBArray[sliceList]-meanB))
    else:
        mask=np.ones(arrayA.shape)
    if type(calFill)==type(None):
        calFill=(np.sum(arrayA*mask)/np.sum(mask)).astype(arrayA.dtype)
    maxTranslate=np.array(arrayB.shape[-len(translateLimit):])*translateLimit
    maxTranslate=np.concatenate((maxTranslate,10.*np.ones(len(initialTranslate)-len(maxTranslate))))
    minTranslate=-maxTranslate
    optimizing_algorithm=gradient_ascent(correlationFunc_translate,initialTranslate,args=(arrayA,arrayB,mask,includeRotate,calFill))
    optimizing_algorithm.maxPara=maxTranslate
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
            newList[nearestIndex]=pointList[n]
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
    newPoints,newCoordRef=pointsToPlanes(points,coordRef)
    newPoints=fillplanesWithPoints(newPoints)
    return newPoints
def reduceResolution(array,lastaxes,nres):
    if nres<2:
        return array.copy()
    else:
        resultArray=array.copy()
        for axisN in range(lastaxes):
            sliceList=[]
            for n in range(len(array.shape)-axisN-1):
                sliceList.append(slice(None))
            sliceList.append(slice(0,array.shape[-1-axisN]-(array.shape[-1-axisN]%nres),nres))
            copyArray=resultArray.copy()
            resultArray=copyArray[tuple(sliceList)]/nres
            for m in range(1,nres):
                sliceList=[]
                for n in range(len(array.shape)-axisN-1):
                    sliceList.append(slice(None))
                sliceList.append(slice(m,array.shape[-1-axisN]-(array.shape[-1-axisN]%nres),nres))
                resultArray+=copyArray[tuple(sliceList)]/nres
    return resultArray
'''
external use functions
'''
def alignAxes_translate(image,axesToTranslate,refAxis,dimSlice=None,fixedRef=False,initTranslate=True,translateLimit=0.5,includeRotate=False,calFill=0,mask=None,nres=1):
    '''refAxis={'axis':index}'''
    if isinstance(translateLimit,float):
        translateLimit=np.ones(len(axesToTranslate))*translateLimit
    trlen=len(axesToTranslate)
    if type(dimSlice)==type(None):
        dimSlice={}
    image=image.clone()
    returnDim=image.dim[:]
    axisref=list(refAxis.keys())[0]
    indexref=refAxis[axisref]
    rearrangeMask=np.arange(len(image.dim)-1)
    rearrangeMask=np.insert(rearrangeMask,image.dim.index(axisref),-1)
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
    rearrange1=image.rearrangeDim(dimensionkey,True)
    rearrange2=image.rearrangeDim(axesToTranslate,False)
    rearrangeMask=rearrangeMask[rearrange1]
    rearrangeMask=rearrangeMask[rearrange2]
    rearrangeMask=rearrangeMask[rearrangeMask>=0]
    if type(mask)!=type(None):
        if type(mask) in [list,tuple]:
            mask=list(np.array(mask)[rearrangeMask])
        elif type(mask)==np.ndarray:
            mask=np.transpose(mask,rearrangeMask)
      
    extractArray=np.copy(image.data[tuple(dimensionSlice)])
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
        logger.info('running slice '+str(n-1))
        if fixedRef:
            ref=relativeIndexref
            indSlice=slice(n-1,n)
            if not(initTranslate):
                translateIndex=None
        else:
            ref=n
            translateIndex=None
            if includeRotate:
                indSlice=slice(0,n)
            else:
                indSlice=slice(n-1,n)
                if initTranslate and len(saveTranslateIndex)>0:
                    translateIndex=-np.array(saveTranslateIndex)[:,1:].sum(axis=0)
        for m in range(nres,0,-1):
            fixArray=reduceResolution(extractArray[ref],trlen,m)
            movArray=reduceResolution(extractArray[n-1],trlen,m)
            if m>1 and type(mask)!=type(None):
                if type(mask)==list:
                    nresMask=[]
                    for naxis in range(len(mask)):
                        nresMask.append((mask[naxis]/m).astype(mask[naxis].dtype))
                elif type(mask)==np.ndarray:
                    nresMask=reduceResolution(mask,trlen,m)
            else:
                nresMask=mask
            if m>1 and type(translateIndex)!=type(None):
                translateIndex[:trlen]/=m
            translateIndex=correlation_translate(fixArray,movArray,translateLimit,initialTranslate=translateIndex,includeRotate=includeRotate,calFill=calFill,mask=nresMask)
            if m>1:
                translateIndex[:trlen]*=m
        if (np.abs(translateIndex)>=0.5).any() or (includeRotate and np.any(np.abs(translateIndex)[trlen:]>0.05)):
            logger.info('updating... with translation '+str(translateIndex))
            saveTranslateIndex.append([n-1,*translateIndex])
            if fixedRef or includeRotate:
                extractArray[indSlice]=translateArray(extractArray[indSlice],translateIndex,includeRotate,0)
    if not(fixedRef) and not(includeRotate):
        for n in range(len(saveTranslateIndex)):
            extractArray[saveTranslateIndex[n][0]]=translateArray(extractArray[saveTranslateIndex[n][0]],np.array(saveTranslateIndex)[:n,1:].sum(axis=0),includeRotate,0)
    nextToTranslate=len(saveTranslateIndex)
    translateIndex=None
    for n in range(relativeIndexref,len(extractArray)-1):
        logger.info('running slice '+str(n+1))
        if fixedRef:
            ref=relativeIndexref
            indSlice=slice(n+1,n+2)
            if not(initTranslate):
                translateIndex=None
        else:
            ref=n
            translateIndex=None
            if includeRotate:
                indSlice=slice(n+1,None)
            else:
                indSlice=slice(n+1,n+2)
                if initTranslate and len(saveTranslateIndex)>nextToTranslate:
                    translateIndex=-np.array(saveTranslateIndex)[nextToTranslate:,1:].sum(axis=0)
        for m in range(nres,0,-1):
            fixArray=reduceResolution(extractArray[ref],trlen,m)
            movArray=reduceResolution(extractArray[n+1],trlen,m)
            if m>1 and type(mask)!=type(None):
                if type(mask)==list:
                    nresMask=[]
                    for naxis in range(len(mask)):
                        nresMask.append((mask[naxis]/m).astype(mask[naxis].dtype))
                elif type(mask)==np.ndarray:
                    nresMask=reduceResolution(mask,trlen,m)
            else:
                nresMask=mask
            if m>1 and type(translateIndex)!=type(None):
                translateIndex[:trlen]/=m
            translateIndex=correlation_translate(fixArray,movArray,translateLimit,initialTranslate=translateIndex,includeRotate=includeRotate,calFill=calFill,mask=nresMask)
            if m>1:
                translateIndex[:trlen]*=m
        if (np.abs(translateIndex)>=0.5).any() or (includeRotate and np.any(np.abs(translateIndex)[trlen:]>0.05)):
            logger.info('updating... with translation '+str(translateIndex))
            saveTranslateIndex.append([n+1,*translateIndex])
            if fixedRef or includeRotate:
                extractArray[indSlice]=translateArray(extractArray[indSlice],translateIndex,includeRotate,0)
    if not(fixedRef) and not(includeRotate):
        for n in range(nextToTranslate,len(saveTranslateIndex)):
            extractArray[saveTranslateIndex[n][0]]=translateArray(extractArray[saveTranslateIndex[n][0]],np.array(saveTranslateIndex)[nextToTranslate:n,1:].sum(axis=0),includeRotate,0)
    if len(saveTranslateIndex)==0:
        saveTranslateIndex=np.zeros((0,len(translateIndex)))
    else:
        saveTranslateIndex=np.array(saveTranslateIndex)
    image.data[dimensionSlice]=np.copy(extractArray)
    image.rearrangeDim(returnDim,True)
    return (image,np.array(saveTranslateIndex))
class alignAxesClass:
    def __init__(self,image,axesToTranslate,refAxis,dimSlice={},includeRotate=False,mask=None,motionOrder=2,temporalSmoothing=3):
        self.temporalSmoothing=temporalSmoothing
        self.motionOrder=motionOrder
        self.image=image.clone()
        self.axesToTranslate=axesToTranslate
        self.dimSlice=dimSlice
        self.includeRotate=includeRotate
        self.mask=mask
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
        self.dimensionSlice=[axisSlice]
        for dimension in dimSlice:
            dimensionkey.append(dimension)
            self.dimensionSlice.append(dimSlice[dimension])
        self.image.rearrangeDim(dimensionkey,True)
        self.image.rearrangeDim(axesToTranslate,False)
        
        self.extractArray=np.copy(self.image.data[tuple(self.dimensionSlice)])
        if axisSlice.start is None:
            axisSliceStart=0
        else:
            axisSliceStart=axisSlice.start
        if axisSlice.step is None:
            axisSliceStep=1
        else:
            axisSliceStep=axisSlice.step   
        self.relativeIndexref=int((indexref-axisSliceStart)/axisSliceStep)
    def predict(self,translateList,*argsNotUsed,**kargsNotUsed):
        translateList=np.array(translateList)
        NumberOfData=(self.motionOrder+1)*self.temporalSmoothing
        ref=self.relativeIndexref
        n=len(translateList)
        if n>ref:
            translateList=np.insert(translateList,ref+1,translateList[0],axis=0)
        if len(translateList)<NumberOfData:
            data=np.concatenate((np.repeat(translateList[:1],NumberOfData-len(translateList),axis=0),translateList[:],),axis=0)
        else:
            data=translateList[-NumberOfData:]
        polynomials = np.polyfit(np.arange(-NumberOfData,0), data, self.motionOrder,full=True)
        prediction=polynomials[0][-1]
        sumSq=polynomials[1]
        return (prediction,np.array(sumSq).sum()/NumberOfData+1.)
    def trackNext(self,translateList,currenttranslate,*argsNotUsed,**kargsNotUsed):
        ref=self.relativeIndexref
        n=len(translateList)
        if n<=ref:
            n=n-1
        logger.info('running slice '+str(n))
        translateIndex=correlation_translate(self.extractArray[ref],self.extractArray[n],np.ones(len(self.axesToTranslate))*0.5,initialTranslate=currenttranslate,includeRotate=self.includeRotate,mask=self.mask)
        return (translateIndex,1)
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
                logger.warning('Calculation error, Input points again.')
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
            logger.info('Registering t '+str(n+1)+' wrt t '+str(n))
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOn()
            elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(np.copy(image.data[n]), isVector=False))
            elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=False))
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            if verifyImg:
                sitk.WriteImage(elastixImageFilter.GetResultImage(),pointfile+'Output/resultImgnb'+str(n+1)+'.mha')
            logger.info('Transforming t '+str(n)+' to t '+str(n+1))
            transformixImageFilter=sitk.TransformixImageFilter()
            transformixImageFilter.LogToFileOn()
            transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
            transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=False))
            transformixImageFilter.SetFixedPointSetFileName(pointfile+'Output/outputpoints.pts')
            transformixImageFilter.SetOutputDirectory(pointfile+'Output')
            transformixImageFilter.Execute()
            newPixelVertice_neighbour=convertElastxOutputToNumpy(pointfile+'Output/outputpoints.txt')
        if baseRefFraction!=0. or n==0:
            logger.info('Registering t '+str(n+1)+' wrt t '+str(0))
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOn()
            elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(np.copy(image.data[0]), isVector=False))
            elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=False))
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            if verifyImg:
                sitk.WriteImage(elastixImageFilter.GetResultImage(),pointfile+'Output/resultImgbase'+str(n+1)+'.mha')
            logger.info('Transforming t '+str(0)+' to t '+str(n+1))
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
        logger.info('Writing STL file to '+pointfile+'_t'+str(n+1)+'.stl')
        
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
            logger.info('Registering t '+str(n+1)+' wrt t '+str(n))
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOn()
            elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(np.copy(image.data[n]), isVector=colorVec))
            elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.copy(image.data[n+1]), isVector=colorVec))
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            logger.info('Transforming t '+str(n)+' to t '+str(n+1))
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
            logger.info('Registering t '+str(n+1)+' wrt t '+str(0))
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
            logger.info('Transforming t '+str(0)+' to t '+str(n+1))
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
        logger.info('Writing MAT file to '+pointfile+str(n+1)+'.mat')
        sio.savemat(pointfile+str(n+1)+'.mat',{('registeredVector'+str(n+1)):vectorMap})
        if baseRefFraction!=1.:
            np.savetxt(pointfile+'Output/outputpoints.pts',newPixelVertice,header='point\n'+str(len(pixelVertice)),comments='')

def TmapRegister(image,savePath='',origin=(0.,0.,0.),bgrid=2.,bweight=1.,rms=False,startTime=0,scaleImg=1.,maskArray=None,writeImg=False,twoD=False,nres =3,smoothing=True,cyclic=False):
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
    np.savetxt(savePath+'/transform/fileScale.txt',[scaleImg])
    for n in range(startTime,image.data.shape[0]-1):
        logger.info('Registering t '+str(n+1)+' wrt t '+str(n))
        elastixImageFilter=sitk.ElastixImageFilter()
        elastixImageFilter.LogToFileOff()
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
        if maskArray is not None:
            fixMask=sitk.GetImageFromArray(maskArray[n].astype('uint8'), isVector=colorVec)
            fixMask.SetOrigin(origin)
            fixMask.SetSpacing(spacing)
            movMask=sitk.GetImageFromArray(maskArray[n+1].astype('uint8'), isVector=colorVec)
            movMask.SetOrigin(origin)
            movMask.SetSpacing(spacing)
            elastixImageFilter.SetFixedMask(fixMask)
            elastixImageFilter.SetMovingMask(movMask)
        elastixImageFilter.Execute()
        Tmap=elastixImageFilter.GetTransformParameterMap()
        for m in range(len(Tmap)):
            sitk.WriteParameterFile(Tmap[m],savePath+'/transform/t'+str(n)+'to'+str(n+1)+'_'+str(m)+'.txt')
        if writeImg:
            sitk.WriteImage(elastixImageFilter.GetResultImage(),savePath+'/t'+str(n)+'to'+str(n+1)+'_resultImg.mha')
        if cyclic:
            logger.info('Registering t '+str(n)+' wrt t '+str(n+1))
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOff()
            elastixImageFilter.LogToConsoleOff()
            elastixImageFilter.SetFixedImage(movImg)
            elastixImageFilter.SetMovingImage(fixImg)
            elastixImageFilter.SetParameterMap(parameterMapVector)
            if maskArray is not None:
                elastixImageFilter.SetFixedMask(movMask)
                elastixImageFilter.SetMovingMask(fixMask)
            elastixImageFilter.Execute()
            Tmap=elastixImageFilter.GetTransformParameterMap()
            for m in range(len(Tmap)):
                sitk.WriteParameterFile(Tmap[m],savePath+'/transform/t'+str(n+1)+'to'+str(n)+'_'+str(m)+'.txt')
            if writeImg:
                sitk.WriteImage(elastixImageFilter.GetResultImage(),savePath+'/t'+str(n+1)+'to'+str(n)+'_resultImg.mha')
        elif n!=0:
            logger.info('Registering t '+str(n+1)+' wrt t '+str(0))
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOff()
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
            if maskArray is not None:
                fixMask=sitk.GetImageFromArray(maskArray[0].astype('uint8'), isVector=colorVec)
                fixMask.SetOrigin(origin)
                fixMask.SetSpacing(spacing)
                movMask=sitk.GetImageFromArray(maskArray[n+1].astype('uint8'), isVector=colorVec)
                movMask.SetOrigin(origin)
                movMask.SetSpacing(spacing)
                elastixImageFilter.SetFixedMask(fixMask)
                elastixImageFilter.SetMovingMask(movMask)
            elastixImageFilter.Execute()
            Tmap=elastixImageFilter.GetTransformParameterMap()
            for m in range(len(Tmap)):
                sitk.WriteParameterFile(Tmap[m],savePath+'/transform/t0to'+str(n+1)+'_'+str(m)+'.txt')
            if writeImg:
                sitk.WriteImage(elastixImageFilter.GetResultImage(),savePath+'/t0to'+str(n+1)+'_resultImg.mha')
    if cyclic:
        logger.info('Registering t '+str(0)+' wrt t '+str(image.data.shape[0]-1))
        elastixImageFilter=sitk.ElastixImageFilter()
        elastixImageFilter.LogToFileOff()
        elastixImageFilter.LogToConsoleOff()
        fixImg=sitk.GetImageFromArray(np.copy(image.data[-1]), isVector=colorVec)
        fixImg.SetOrigin(origin)
        fixImg.SetSpacing(spacing)
        movImg=sitk.GetImageFromArray(np.copy(image.data[0]), isVector=colorVec)
        movImg.SetOrigin(origin)
        movImg.SetSpacing(spacing)
        elastixImageFilter.SetFixedImage(fixImg)
        elastixImageFilter.SetMovingImage(movImg)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        if maskArray is not None:
            fixMask=sitk.GetImageFromArray(maskArray[-1].astype('uint8'), isVector=colorVec)
            fixMask.SetOrigin(origin)
            fixMask.SetSpacing(spacing)
            movMask=sitk.GetImageFromArray(maskArray[0].astype('uint8'), isVector=colorVec)
            movMask.SetOrigin(origin)
            movMask.SetSpacing(spacing)
            elastixImageFilter.SetFixedMask(fixMask)
            elastixImageFilter.SetMovingMask(movMask)
        elastixImageFilter.Execute()
        Tmap=elastixImageFilter.GetTransformParameterMap()
        for m in range(len(Tmap)):
            sitk.WriteParameterFile(Tmap[m],savePath+'/transform/t'+str(image.data.shape[0]-1)+'to'+str(0)+'_'+str(m)+'.txt')
        if writeImg:
            sitk.WriteImage(elastixImageFilter.GetResultImage(),savePath+'/t'+str(image.data.shape[0]-1)+'to'+str(0)+'_resultImg.mha')
        logger.info('Registering t '+str(image.data.shape[0]-1)+' wrt t '+str(0))
        elastixImageFilter=sitk.ElastixImageFilter()
        elastixImageFilter.LogToFileOff()
        elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.SetFixedImage(movImg)
        elastixImageFilter.SetMovingImage(fixImg)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.Execute()
        Tmap=elastixImageFilter.GetTransformParameterMap()
        for m in range(len(Tmap)):
            sitk.WriteParameterFile(Tmap[m],savePath+'/transform/t'+str(0)+'to'+str(image.data.shape[0]-1)+'_'+str(m)+'.txt')
        if writeImg:
            sitk.WriteImage(elastixImageFilter.GetResultImage(),savePath+'/t'+str(0)+'to'+str(image.data.shape[0]-1)+'_resultImg.mha')
def TmapRegister_img2img(image1,image2,savePath='',fileName='img2img',scaleImg=1.,tInd=None,origin1=(0.,0.,0.),origin2=(0.,0.,0.),maskArray1=None,maskArray2=None,EulerTransformCorrection=False,rms=False,bgrid=2.,bweight=1.,twoD=False,nres =3,smoothing=True):
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
        elastixImageFilter.LogToFileOff()
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
        if maskArray1 is not None:
            fixMask=sitk.GetImageFromArray(maskArray1.astype('uint8'), isVector=colorVec)
            fixMask.SetOrigin(origin1)
            fixMask.SetSpacing(spacing1)
            elastixImageFilter.SetFixedMask(fixMask)
        if maskArray2 is not None:
            movMask=sitk.GetImageFromArray(maskArray2.astype('uint8'), isVector=colorVec)
            movMask.SetOrigin(origin2)
            movMask.SetSpacing(spacing2)
            elastixImageFilter.SetMovingMask(movMask)
        elastixImageFilter.Execute()
        Tmap=elastixImageFilter.GetTransformParameterMap()
        for m in range(len(Tmap)):
            sitk.WriteParameterFile(Tmap[m],savePath+'/transform/'+fileName+'.txt')
    else:
        for n in tInd:
            logger.info('Registering t '+str(n))
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOff()
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
            if maskArray1 is not None:
                fixMask=sitk.GetImageFromArray(maskArray1[n].astype('uint8'), isVector=colorVec)
                fixMask.SetOrigin(origin1)
                fixMask.SetSpacing(spacing1)
                elastixImageFilter.SetFixedMask(fixMask)
            if maskArray2 is not None:
                movMask=sitk.GetImageFromArray(maskArray2[n].astype('uint8'), isVector=colorVec)
                movMask.SetOrigin(origin2)
                movMask.SetSpacing(spacing2)
                elastixImageFilter.SetMovingMask(movMask)
            elastixImageFilter.Execute()
            Tmap=elastixImageFilter.GetTransformParameterMap()
            for m in range(len(Tmap)):
                sitk.WriteParameterFile(Tmap[m],savePath+'/transform/'+fileName+'_t'+str(n)+'.txt')
def nonRigidRegistration(nonRigidSavePath,imageArray,gridSize,bgridFactor=1,metric='AdvancedMeanSquares',inverse=True,full=True):

    os.makedirs(nonRigidSavePath+'/transform', exist_ok=True)

    origin=tuple(np.zeros(len(imageArray.shape)-1))
    spacing=tuple(np.ones(len(imageArray.shape)-1))
    parameterMapVector = sitk.VectorOfParameterMap() 
    bspline=sitk.GetDefaultParameterMap("bspline")
    bspline["AutomaticScalesEstimation"]=( "true", )
    bspline["AutomaticTransformInitialization"] = ( "true", )
    bspline['Metric0Weight']=(str(1),)
    bspline['FinalGridSpacingInPhysicalUnits']=tuple((gridSize*bgridFactor).astype(str))
    bspline["Metric"]=(metric,bspline["Metric"][1])

    parameterMapVector.append(bspline)

    '''start'''
    firstCyclicTimeStep=imageArray.shape[0]
    colorVec=False
    for nn in range(firstCyclicTimeStep):
        fixImg=sitk.GetImageFromArray(np.copy(imageArray[nn]), isVector=colorVec)
        fixImg.SetOrigin(origin)
        fixImg.SetSpacing(spacing)
        for n in range(imageArray.shape[0]):
            if n==nn:
                continue
            if not(full):
                if n==(imageArray.shape[0]-1) and nn==0:
                    pass
                elif n==0 and nn==(imageArray.shape[0]-1):
                    pass
                elif abs(n-nn)>1:
                    continue
            if inverse:
                filename='tstep'+str(n)+'to'+str(nn)
            else:
                filename='tstep'+str(nn)+'to'+str(n)
            logger.info('Registering '+filename)
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOff()
            elastixImageFilter.LogToConsoleOff()
            movImg=sitk.GetImageFromArray(np.copy(imageArray[n]), isVector=colorVec)
            movImg.SetOrigin(origin)
            movImg.SetSpacing(spacing)
            if inverse:
                elastixImageFilter.SetFixedImage(movImg)
                elastixImageFilter.SetMovingImage(fixImg)
            else:
                elastixImageFilter.SetFixedImage(fixImg)
                elastixImageFilter.SetMovingImage(movImg)
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            Tmap=elastixImageFilter.GetTransformParameterMap()
            for m in range(len(Tmap)):
                sitk.WriteParameterFile(Tmap[m],nonRigidSavePath+'/transform/'+filename+'_'+str(m)+'.txt')
    return;
def cyclicNonRigidCorrection(cyclicTimeStep,imageArray,gridSize,nonRigidSavePath='',bgridFactor=1,metric='AdvancedMeanSquares',inverse=False,returnSyncPhase=False,saveImage=False):
    if nonRigidSavePath:
        os.makedirs(nonRigidSavePath+'/transform', exist_ok=True)
    if isinstance(cyclicTimeStep,list):
        firstCyclicTimeStep=cyclicTimeStep[1]
        cyclicSlice=[]
        cyc_temp=np.array(cyclicTimeStep)[1:]-np.array(cyclicTimeStep)[:-1]
        for n in range(firstCyclicTimeStep):
            phase=n/float(cyclicTimeStep[1])
            cyclicSlice.append(np.around(phase*cyc_temp+np.array(cyclicTimeStep)[:-1]).astype(int))
        cyclicSlice=np.array(cyclicSlice)
    elif isinstance(cyclicTimeStep,int):
        firstCyclicTimeStep=cyclicTimeStep
        cyclicSlice=[]
        for n in range(firstCyclicTimeStep):
            cyclicSlice.append(np.arange(n,int(imageArray.shape[0]/cyclicTimeStep)*cyclicTimeStep,cyclicTimeStep).astype(int))
        cyclicSlice=np.array(cyclicSlice)
    else:
        firstCyclicTimeStep=cyclicTimeStep.shape[0]
        cyclicSlice=cyclicTimeStep.copy()
    origin=tuple(np.zeros(len(imageArray.shape)-1))
    spacing=tuple(np.ones(len(imageArray.shape)-1))

    parameterMapVector = sitk.VectorOfParameterMap() 
    bspline=sitk.GetDefaultParameterMap("bspline")
    bspline["AutomaticScalesEstimation"]=( "true", )
    bspline["AutomaticTransformInitialization"] = ( "true", )
    bspline['Metric0Weight']=(str(1),)
    bspline['FinalGridSpacingInPhysicalUnits']=tuple((gridSize*bgridFactor).astype(str))
    bspline["Metric"]=(metric,bspline["Metric"][1])

    parameterMapVector.append(bspline)

    newImageArray=np.zeros((firstCyclicTimeStep*cyclicTimeStep.shape[1],*imageArray.shape[1:]))
    '''start'''
    colorVec=False
    for nn in range(firstCyclicTimeStep):
        tempImageArray=imageArray[cyclicSlice[nn]].copy()
        fixImg=sitk.GetImageFromArray(np.copy(tempImageArray[0]), isVector=colorVec)
        fixImg.SetOrigin(origin)
        fixImg.SetSpacing(spacing)
        for n in range(1,len(tempImageArray)):
            if inverse:
                filename='t'+str(cyclicSlice[nn][n])+'to'+str(cyclicSlice[nn][0])
            else:
                filename='t'+str(cyclicSlice[nn][0])+'to'+str(cyclicSlice[nn][n])
            logger.info('Registering '+filename)
            elastixImageFilter=sitk.ElastixImageFilter()
            elastixImageFilter.LogToFileOff()
            elastixImageFilter.LogToConsoleOff()
            movImg=sitk.GetImageFromArray(np.copy(tempImageArray[n]), isVector=colorVec)
            movImg.SetOrigin(origin)
            movImg.SetSpacing(spacing)
            if inverse:
                elastixImageFilter.SetFixedImage(movImg)
                elastixImageFilter.SetMovingImage(fixImg)
            else:
                elastixImageFilter.SetFixedImage(fixImg)
                elastixImageFilter.SetMovingImage(movImg)
            elastixImageFilter.SetParameterMap(parameterMapVector)
            elastixImageFilter.Execute()
            Tmap=elastixImageFilter.GetTransformParameterMap()
            if nonRigidSavePath:
                for m in range(len(Tmap)):
                    sitk.WriteParameterFile(Tmap[m],nonRigidSavePath+'/transform/'+filename+'_'+str(m)+'.txt')
                if saveImage:
                    sitk.WriteImage(elastixImageFilter.GetResultImage(),nonRigidSavePath+'/'+filename+'_resultImg.mha')
            tempImageArray[n]=sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
        newImageArray[nn::cyclicTimeStep.shape[0]]=tempImageArray.copy()
    if returnSyncPhase:
        return (newImageArray,cyclicSlice)
    else:
        return newImageArray
def TmapRegister_rigid(image1,image2,savePath='',fileInit=None,fileName='img2img',origin1=(0.,0.,0.),origin2=(0.,0.,0.),maskArray1=None,maskArray2=None,bsplineTransformCorrection=False,rms=True,bgrid=2.,bweight=1.):
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
        logger.info('Set initial transform parameters: '+str(fileInit))
    elastixImageFilter.SetParameterMap(parameterMapVector)
    if maskArray1 is not None:
        fixMask=sitk.GetImageFromArray(maskArray1.astype('uint8'), isVector=colorVec)
        fixMask.SetOrigin(origin1)
        fixMask.SetSpacing(spacing1)
        elastixImageFilter.SetFixedMask(fixMask)
    if maskArray2 is not None:
        movMask=sitk.GetImageFromArray(maskArray2.astype('uint8'), isVector=colorVec)
        movMask.SetOrigin(origin2)
        movMask.SetSpacing(spacing2)
        elastixImageFilter.SetMovingMask(movMask)
    elastixImageFilter.Execute()
    Tmap=elastixImageFilter.GetTransformParameterMap()
    if os.path.isfile(fileInit):
        for m in range(len(Tmap)):
            sitk.WriteParameterFile(Tmap[m],savePath+'/transform/'+fileName+'_'+str(m)+'.txt')
    else:
        sitk.WriteParameterFile(Tmap[0],fileInit)
    sitk.WriteImage(elastixImageFilter.GetResultImage(),savePath+'/'+fileName+'_resultImg.mha')

def transform(stlFile,timeStepNo,mapNo,startTime=0,cumulative=True,ratioFunc=timestepPoly,savePath='',TmapPath='',scale=1.,delimiter=' ',forwardbackward=False):
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
    if forwardbackward:
        addSaveStr='_forwardbackward'
        np.savetxt(savePath+'/input.pts',oriPos,header='point\n'+str(len(oriPos)),comments='')
        forward=[]
        for n in range(timeStepNo-1):
            fromTime=startTime+n
            if fromTime>=timeStepNo:
                fromTime-=timeStepNo
            toTime=fromTime+1
            if toTime>=timeStepNo:
                toTime-=timeStepNo
            Tmap=[]#SimpleITK.VectorOfParameterMap()
            for m in range(mapNo):
                Tmap.append(sitk.ReadParameterFile(TmapPath+'/transform/t'+str(fromTime)+'to'+str(toTime)+'_'+str(m)+'.txt'))

            transformixImageFilter=sitk.TransformixImageFilter()
            transformixImageFilter.LogToFileOff()
            transformixImageFilter.LogToConsoleOff()
            transformixImageFilter.SetTransformParameterMap(Tmap)
            transformixImageFilter.SetMovingImage(sitk.ReadImage(TmapPath+'/t0Img.mha'))
            transformixImageFilter.SetFixedPointSetFileName(savePath+'/input.pts')
            transformixImageFilter.SetOutputDirectory(savePath)
            transformixImageFilter.Execute()

            with open (savePath+'/outputpoints.txt', "r") as myfile:
                data=myfile.readlines()
            newPos=[]
            for string in data:
                result = re.search('OutputPoint(.*)Deformation', string)
                newPos.append(np.fromstring(result.group(1)[5:-6], sep=' '))
            newPos=np.array(newPos)
            forward.append(newPos.copy())
            np.savetxt(savePath+'/input.pts',newPos,header='point\n'+str(len(newPos)),comments='')
        np.savetxt(savePath+'/input.pts',oriPos,header='point\n'+str(len(oriPos)),comments='')
        backward=[]
        for n in range(timeStepNo-1):
            fromTime=startTime-n
            if fromTime<0:
                fromTime+=timeStepNo
            toTime=fromTime-1
            if toTime<0:
                toTime+=timeStepNo
            Tmap=[]#SimpleITK.VectorOfParameterMap()
            for m in range(mapNo):
                Tmap.append(sitk.ReadParameterFile(TmapPath+'/transform/t'+str(fromTime)+'to'+str(toTime)+'_'+str(m)+'.txt'))

            transformixImageFilter=sitk.TransformixImageFilter()
            transformixImageFilter.LogToFileOff()
            transformixImageFilter.LogToConsoleOff()
            transformixImageFilter.SetTransformParameterMap(Tmap)
            transformixImageFilter.SetMovingImage(sitk.ReadImage(TmapPath+'/t0Img.mha'))
            transformixImageFilter.SetFixedPointSetFileName(savePath+'/input.pts')
            transformixImageFilter.SetOutputDirectory(savePath)
            transformixImageFilter.Execute()

            with open (savePath+'/outputpoints.txt', "r") as myfile:
                data=myfile.readlines()
            newPos=[]
            for string in data:
                result = re.search('OutputPoint(.*)Deformation', string)
                newPos.append(np.fromstring(result.group(1)[5:-6], sep=' '))
            newPos=np.array(newPos)
            backward.append(newPos.copy())
            np.savetxt(savePath+'/input.pts',newPos,header='point\n'+str(len(newPos)),comments='')
        ratio=1./(1.+np.arange(1,timeStepNo)/np.arange(timeStepNo-1,0,-1))
        backward=backward[::-1]
        for n in range(timeStepNo-1):
            toTime=startTime+n+1
            if toTime>=timeStepNo:
                toTime-=timeStepNo
            newPos=forward[n]*ratio[n]+(1-ratio[n])*backward[n]
            if stlFile[-3:]=='stl':
                ref_mesh.vertices=np.array(newPos)*scale
                trimesh.io.export.export_mesh(ref_mesh,savePath+'/t'+str(toTime)+addSaveStr+'.stl')
            else:
                np.savetxt(savePath+'/t'+str(toTime)+addSaveStr+'.txt',np.array(newPos)*scale)
    else:
        for n in range(timeStepNo-1):
            fromTime=startTime+n
            if fromTime>=timeStepNo:
                fromTime-=timeStepNo
            toTime=fromTime+1
            if toTime>=timeStepNo:
                toTime-=timeStepNo
            Tmap=[]#SimpleITK.VectorOfParameterMap()
            for m in range(mapNo):
                Tmap.append(sitk.ReadParameterFile(TmapPath+'/transform/t0to'+str(toTime)+'_'+str(m)+'.txt'))

            transformixImageFilter=sitk.TransformixImageFilter()
            transformixImageFilter.LogToFileOff()
            transformixImageFilter.LogToConsoleOff()
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
                if n>0 and n<(timeStepNo-2):
                    ratio=ratioFunc(n+1,timeStepNo)
                    Tmap=[]#SimpleITK.VectorOfParameterMap()
                    for m in range(mapNo):
                        Tmap.append(sitk.ReadParameterFile(TmapPath+'/transform/t'+str(fromTime)+'to'+str(toTime)+'_'+str(m)+'.txt'))

                    transformixImageFilter=sitk.TransformixImageFilter()
                    transformixImageFilter.LogToFileOff()
                    transformixImageFilter.LogToConsoleOff()
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
                trimesh.io.export.export_mesh(ref_mesh,savePath+'/t'+str(toTime)+addSaveStr+'.stl')
            else:
                np.savetxt(savePath+'/t'+str(toTime)+addSaveStr+'.txt',np.array(newPos)*scale)
def transform_img2img(stlFile,trfFile,savePath='',mhaFile='',fileName='trf',scale=1.,delimiter=' '):
    if savePath=='':
        savePath=stlFile[:-4]
    os.makedirs(savePath, exist_ok=True)
    if type(stlFile)!=str:
        oriPos=stlFile.copy()
    elif stlFile[-3:]=='stl':
        ref_mesh=trimesh.load(stlFile)
        oriPos=np.array(ref_mesh.vertices)
    else:
        oriPos=np.loadtxt(stlFile,delimiter=delimiter)
    
    if oriPos.shape[-1]>5 or len(oriPos.shape)!=2:
        trfImage=True
    else:
        trfImage=False
        oriPos=oriPos/scale
        np.savetxt(savePath+'/input0.pts',oriPos,header='point\n'+str(len(oriPos)),comments='')
    Tmap=[]#SimpleITK.VectorOfParameterMap()
    Tmap.append(sitk.ReadParameterFile(trfFile))
    
    transformixImageFilter=sitk.TransformixImageFilter()
    transformixImageFilter.LogToFileOff()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetTransformParameterMap(Tmap)
    if trfImage:
        movImg=sitk.GetImageFromArray(oriPos, isVector=False)
        movImg.SetSpacing(np.ones(len(oriPos.shape))*scale)
        transformixImageFilter.SetMovingImage(movImg)
    else:
        if os.path.isfile(mhaFile):
            transformixImageFilter.SetMovingImage(sitk.ReadImage(mhaFile))
        elif os.path.isfile(savePath+'/t0Img.mha'):
            transformixImageFilter.SetMovingImage(sitk.ReadImage(savePath+'/t0Img.mha'))
        else:
            transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(np.zeros(np.ceil(oriPos.max(axis=0)*1.1).astype(int)[::-1]), isVector=False))
        transformixImageFilter.SetFixedPointSetFileName(savePath+'/input0.pts')
    transformixImageFilter.SetOutputDirectory(savePath)
    transformixImageFilter.Execute()
    if trfImage:
        newPos=sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())
        sitk.WriteImage(transformixImageFilter.GetResultImage(),savePath+'/'+fileName+'.mha')
    else:
        with open (savePath+'/outputpoints.txt', "r") as myfile:
            data=myfile.readlines()
        newPos=[]
        for string in data:
            result = re.search('OutputPoint(.*)Deformation', string)
            newPos.append(np.fromstring(result.group(1)[5:-6], sep=' '))
        newPos=np.array(newPos)
        if stlFile[-3:]=='stl':
            ref_mesh.vertices=newPos*scale
            trimesh.io.export.export_mesh(ref_mesh,savePath+'/'+fileName+'.stl')
        else:
            np.savetxt(savePath+'/'+fileName+'.txt',newPos*scale)
        newPos=newPos*scale
    return newPos
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
            logger.info('Registering t '+str(n)+' wrt t '+str(n+1))
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
        
        logger.info('Registering t '+str(0)+' wrt t '+str(n+1))
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
        logger.info(tPath)
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
        
        
    def register(self,image1,image2,initialTransf='',savePath='',regType='rigid',fileName='img2img',metric='rms',nres=6,smoothing=True,outputImage=True,maskArray1=None,maskArray2=None,elastixLog=False):
        if type(image1)==np.ndarray:
            twoD=len(image1.shape)
            origin1=tuple(np.zeros(twoD))
            spacing1=tuple(np.ones(twoD))
            origin2=tuple(np.zeros(twoD))
            spacing2=tuple(np.ones(twoD))
        else:
            twoD=3
            image1=image1.clone()
            image2=image2.clone()
            image1.rearrangeDim(['z','y','x'])
            image2.rearrangeDim(['z','y','x'])
            origin1=(0.,0.,0.)
            origin2=(0.,0.,0.)
            spacing1=(image1.dimlen['x'],image1.dimlen['y'],image1.dimlen['z'])
            spacing2=(image2.dimlen['x'],image2.dimlen['y'],image2.dimlen['z'])
        
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
        
        parameterMapVector = sitk.VectorOfParameterMap()
        EulerTransform=sitk.GetDefaultParameterMap("rigid")
        EulerTransform["AutomaticScalesEstimation"]=( "true", ) 
        EulerTransform["AutomaticTransformInitialization"] = ( "true", )
        if maskArray1 is not None or maskArray2 is not None:
            EulerTransform["ImageSampler"]=["RandomSparseMask"]
        affine=sitk.GetDefaultParameterMap("affine")
        affine["AutomaticScalesEstimation"]=( "true", ) 
        affine["AutomaticTransformInitialization"] = ( "true", ) 
        if maskArray1 is not None or maskArray2 is not None:
            affine["ImageSampler"]=["RandomSparseMask"]
        
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

        EulerTransform["NumberOfResolutions"] = (str(nres),)
        affine["NumberOfResolutions"] = (str(nres),)
        if smoothing:
            EulerTransform["FixedImagePyramidSchedule"] = tuple((2**np.repeat(np.arange(nres-1,-1,-1),twoD)).astype(str))
            EulerTransform["MovingImagePyramidSchedule"] = tuple((2**np.repeat(np.arange(nres-1,-1,-1),twoD)).astype(str))
            affine["FixedImagePyramidSchedule"] = tuple((2**np.repeat(np.arange(nres-1,-1,-1),twoD)).astype(str))
            affine["MovingImagePyramidSchedule"] = tuple((2**np.repeat(np.arange(nres-1,-1,-1),twoD)).astype(str))
        parameterMapVector.append(EulerTransform)
        if regType == 'affine':
            parameterMapVector.append(affine)
        #elif regType == 'rigid':
            #parameterMapVector.append(EulerTransform)
        
        
        colorVec=False
        if 'RGB' in image1.dim:
            colorVec=True
        
        elastixImageFilter=sitk.ElastixImageFilter()
        if elastixLog:
            elastixImageFilter.LogToFileOn()
            elastixImageFilter.LogToConsoleOn()
        else:
            elastixImageFilter.LogToFileOff()
            elastixImageFilter.LogToConsoleOff()
        if type(image1)==np.ndarray:
            x=np.copy(image1)
        else:
            x = np.copy(image1.data)
        fixImg=sitk.GetImageFromArray(x.astype(np.uint8), isVector=colorVec)
        
        fixImg.SetOrigin(origin1)
        fixImg.SetSpacing(spacing1)
        if outputImage:
            sitk.WriteImage(fixImg,savePath+'/'+fileName+'_fixImg.mha')
        if type(image2)==np.ndarray:
            movImg=sitk.GetImageFromArray(np.copy(image2.astype(np.uint8)), isVector=colorVec)
        else:
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
        
        if maskArray1 is not None:
            fixMask=sitk.GetImageFromArray(maskArray1.astype('uint8'), isVector=colorVec)
            fixMask.SetOrigin(origin1)
            fixMask.SetSpacing(spacing1)
            elastixImageFilter.SetFixedMask(fixMask)
        if maskArray2 is not None:
            movMask=sitk.GetImageFromArray(maskArray2.astype('uint8'), isVector=colorVec)
            movMask.SetOrigin(origin2)
            movMask.SetSpacing(spacing2)
            elastixImageFilter.SetMovingMask(movMask)
            
        os.makedirs(savePath+'/'+'logFile', exist_ok=True)
        
        elastixImageFilter.SetLogToFile(True)
        elastixImageFilter.SetOutputDirectory(savePath+'/'+'logFile')
        elastixImageFilter.SetLogFileName('elastix.txt')
        #elastixImageFilter.SetLogToConsole(False)

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
        logger.info(str(output_size))
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
        if self.func is None:
            value=np.array([float('nan')]*val.size)
        val=np.array(val)
        val=val[np.logical_not(np.isnan(val))]
        val=val[val>=1]
        if len(val)==0:
            if self.func is not None:
                if self.returnStats:
                    value=np.array([0,0])
                else:
                    value=0
        elif (val.max()-val.min())<=3:
            if self.func is None:
                value[:val.size]=val.reshape(-1)
            elif self.returnStats:
                value=np.array([self.func(val),0])
            else:
                value=self.func(val)
        else:
            if self.func is not None:
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
                        if self.func is None:
                            value[:val[inliersInd].size]=val[inliersInd].reshape(-1)
                        else:
                            value=self.func(val[inliersInd])
                            if self.returnStats:
                                temp_bincount=bincount.copy()
                                temp_bincount[intensityCompresed[low]:(intensityCompresed[high-1]+1)]=0
                                cs=np.insert(np.cumsum(temp_bincount), 0, 0)
                                cs2=np.roll(cs,-newbound[n,3])
                                value=np.array([value,np.max(cs2-cs)])
                        break
            elif self.func is None:
                value[:val.size]=val.reshape(-1)
            if self.returnStats:
                value=np.array([value,0])
                
        return value
def getPhaseCong(image,saveTo,alpha=0.6,minWaveLength=5,schemeArgs=None):
    logger.debug('Please rearrange the axes in this order [two other axes... , main direction of beam,2D perpendicular direction]')
    logger.debug('Current axis arrangement: '+str(image.dim))
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
            resultImage.data=np.nanmean(image.data,axis=-1)
        else:
            schemeArgs=schemeArgs/np.sum(schemeArgs)
            resultImage.data=np.nansum(image.data*schemeArgs,axis=-1)
    elif scheme=='SAC':
        if type(schemeArgs)==type(None):
            schemeArgs=0.5
            
        SCAfunc=SAC(schemeArgs,returnStats=returnStats)
        image.data=np.minimum(255,np.maximum(0,image.data)).astype('uint8')
        resultData=image.data.reshape((-1,image.data.shape[-1]),order='F')
        if twoD:
            for xn in range(image.data.shape[0]):
                logger.info('    {0:.3f}% completed...'.format(float(xn)/image.data.shape[0]*100.))
                for yn in range(image.data.shape[1]):
                    resultImage.data[xn,yn]=SCAfunc(image.data[xn,yn])
                    #for zn in range(image.data.shape[2]):
                    #    resultImage.data[xn,yn,zn]=SCAfunc(image.data[xn,yn,zn])
        elif not(parallel):
            for xn in range(image.data.shape[0]):
                logger.info('    {0:.3f}% completed...'.format(float(xn)/image.data.shape[0]*100.))
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
        resultImage.data=np.nanmedian(image.data,axis=-1)
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
        for n in range(len(resultImage.data.shape)):
            if resultImage.data.shape[n]!=image.data.shape[n]:
                resultImage.data=resultImage.data[tuple([slice(None)]*n+[slice(image.data.shape[n])])]
    elif scheme=='maximum':
        resultImage.data=np.nanmax(image.data,axis=-1)
    elif scheme=='minimum':
        resultImage.data=np.nanmin(image.data,axis=-1)
    else:
        raise Exception('No Valid Scheme Chosen.')
            
    return resultImage
