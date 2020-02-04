'''
File: PointSpeckleProc.py
Description: Processes for ultrasound point speckles
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         04FEB2020           - Created
  
Requirements:
    numpy
    scipy
Known Bug:
    None
All rights reserved.
'''
_version='2.4.0'
import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
from scipy.ndimage import shift
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from scipy.stats import norm
import processFunc

#Optional dependancies

def gKern(std,kernlen=7,normalize=None):
    """Returns a Gaussian kernel array."""
    if isinstance(kernlen,(float,int)):
        kernlen=(kernlen*np.ones(len(std))).astype(int)
    gkern1d = []
    for n in range(len(std)):
        gkern1d.append(signal.gaussian(kernlen[n]*std, std=std[n]))
    gkern = np.zeros(gkern1d)
    gkern[:]=gkern1d[0].copy()
    gkern=gkern.transpose(np.roll(np.arange(len(std),dtype=int),-1))
    for n in range(1,len(std)):
        gkern[:]*=gkern1d[n]
        gkern=gkern.transpose(np.roll(np.arange(len(std),dtype=int),-1))
    if normalize:
        gkern*=normalize/gkern.max()
    return gkern2d

def getSpecklePoint(image,smoothingSize,prefilter=None):
    newImage=image.clone()
    extraDim=len(image.data.shape)-len(smoothingSize)
    if prefilter=='gaussian':
        tempArray=gaussian_filter(image.data, [*np.zeros(extraDim).astype(int),*smoothingSize],mode='constant')
        i=[]
        adjust=np.mgrid[tuple([slice(-1,2)]*len(smoothingSize))].reshape((2,-1)).T
        for adj in adjust:
            i.append(shift(tempArray,[*np.zeros(extraDim).astype(int),*adj],order=0))
        i=np.array(i)
        newImage.data[np.max(i,axis=0)>tempArray]=0
    else:
        i=[]
        sliceList=[]
        for n in range(len(smoothingSize)):
            sliceList.append(slice(-smoothingSize[n],smoothingSize[n]+1))
        adjust=np.mgrid[tuple(sliceList)].reshape((2,-1)).T
        for adj in adjust:
            i.append(shift(image.data,[*np.zeros(extraDim).astype(int),*adj],order=0))
        i=np.array(i)
        newImage.data[np.max(i,axis=0)>image.data]=0

    #remove duplicate
    nnzInd=np.transpose(np.nonzero(newImage.data))
    for checkInd in nnzInd:
        if newImage.data[tuple(checkInd)]>0:
            currentCheckIndex=checkInd.copy()
            for maxN in range(10):
                sliceList=list(currentCheckIndex[:extraDim])
                addind=np.zeros(len(smoothingSize),dtype=int)
                for nn in range(len(smoothingSize)):
                    sliceList.append(slice(max(0,currentCheckIndex[nn+extraDim]-int(np.ceil(smoothingSize[nn]))),min(newImage.data.shape[nn+extraDim],currentCheckIndex[nn+extraDim]+int(np.ceil(smoothingSize[nn]))+1)))
                    addind[nn]=max(0,currentCheckIndex[nn+extraDim]-int(np.ceil(smoothingSize[nn])))
                repeatedInd=np.transpose(np.nonzero(newImage.data[tuple(sliceList)]))
                if repeatedInd.shape[0]>1:
                    temp=np.around(np.mean(repeatedInd,axis=0)).astype(int)+addind
                    if np.all(temp==currentCheckIndex[extraDim:]):
                        break
                    else:
                        currentCheckIndex[extraDim:]=temp.copy()
                else:
                    break
            sliceList=list(currentCheckIndex[:extraDim])
            emptyslice=False
            for nn in range(len(smoothingSize)):
                sliceList.append(slice(max(0,currentCheckIndex[nn+extraDim]-int(np.ceil(smoothingSize[nn]))),min(newImage.data.shape[nn+extraDim],currentCheckIndex[nn+extraDim]+int(np.ceil(smoothingSize[nn]))+1)))
            temp_avg=newImage.data[tuple(sliceList)][newImage.data[tuple(sliceList)]>0].mean()
            newImage.data[tuple(sliceList)]=0
            newImage.data[tuple(currentCheckIndex)]=temp_avg
    return newImage
def reduceRepeat(image,checkSize,removeSingleton=0):
    reduce=image.data.shape[1]-1
    newImage=image.clone()
    for base in range(image.data.shape[1]):
        ii=[image.data[:,base].astype(bool)]
        for t in range(image.data.shape[1]):
            if t==base:
                continue
            i=[]
            for y in range(-checkSize[0],checkSize[0]+1):
                for x in range(-checkSize[1],checkSize[1]+1):
                            i.append(shift(image.data[:,t],[0,y,x],order=0))
            ii.append(np.any(np.array(i),axis=0))
        ii=np.array(ii).astype(int)
        temp_sum=np.maximum(np.sum(ii,axis=0),1).astype(float)
        if removeSingleton>0:
            newImage.data[:,base][temp_sum<=removeSingleton]=0
        newImage.data[:,base]*=(1-(temp_sum-1)/reduce)/temp_sum
    return newImage
        
def reduceNonRandom(image,sigmas,densityApprox=None,dimSigmaFactor=1.,average=False,truncate=0.5,dim='t',useCorrDet=False):
    oneNzeroArray=image.data.astype(bool).astype(float)
    normVal=(1./gaussian_filter(np.ones(np.ones(len(sigmas)).astype(int)), sigmas,mode='constant')).max()
    imgSigmaData=normVal*gaussian_filter(oneNzeroArray, [*np.zeros(len(image.data.shape)-len(sigmas)).astype(int),*sigmas],mode='constant')
    imgSigmaData=imgSigmaData.sum(axis=1)
    sigmaAvg=np.ones(len(sigmas))*np.mean(sigmas)
    if average:
        normValAvg=(1./gaussian_filter(np.ones(np.ones(len(sigmaAvg)).astype(int)), sigmaAvg,mode='constant',truncate=truncate)).max()
        imgSigmaAvgData=normValAvg*gaussian_filter(oneNzeroArray, [*np.zeros(len(image.data.shape)-len(sigmaAvg)).astype(int),*sigmaAvg],mode='constant',truncate=truncate)
        imgSigmaAvgData=imgSigmaAvgData.sum(axis=1)
    if densityApprox:
        density=densityApprox
    else:
        density=float(oneNzeroArray[oneNzeroArray>0].size)/oneNzeroArray.size
    dimInd=image.dim.index(dim)
    dimIndSlice=[slice(None)]*dimInd

    newImage=image.clone()
    
    if dimSigmaFactor!=0:
        mean=normVal*density*(image.data.shape[dimInd]-1)+1
        std=(mean-1)*abs(dimSigmaFactor)
        
        logger.info('Using mean and std of: {0:.3f} , {1:.3f}'.format(mean,std))
        nProb=norm(mean,std)
        normalize=1./nProb.pdf(mean)
        
        for t in range(image.data.shape[dimInd]):
            if average:
                newImage.data[(*dimIndSlice,t)][newImage.data[(*dimIndSlice,t)]>0]*=normalize*nProb.pdf(imgSigmaData[newImage.data[(*dimIndSlice,t)]>0])/imgSigmaAvgData[newImage.data[(*dimIndSlice,t)]>0]
            elif dimSigmaFactor<0:
                newImage.data[(*dimIndSlice,t)][newImage.data[(*dimIndSlice,t)]>0]=normalize*nProb.pdf(imgSigmaData[newImage.data[(*dimIndSlice,t)]>0])
            else:
                newImage.data[(*dimIndSlice,t)][newImage.data[(*dimIndSlice,t)]>0]*=normalize*nProb.pdf(imgSigmaData[newImage.data[(*dimIndSlice,t)]>0])
        if dimSigmaFactor<0:
            return newImage.data
    if useCorrDet:
        logger.warning('Warning, use only when sample size is large.')
        corrdet=[]
        posALL=[]
        maxCorrdet=0
        for t in range(image.data.shape[dimInd]):
            pos=np.array(np.nonzero(image.data[(*dimIndSlice,t)]>0))
            posALL.append(pos.copy())
            corrdet.append(np.zeros(pos.shape[1]))
            for n in range(pos.shape[1]):
                nearbyPosInd=np.nonzero(np.all(np.logical_and(pos[-2:]>=(pos[-2:,n]-np.ceil(sigmaAvg*2)).reshape((-1,1)),pos[-2:]<=(pos[-2:,n]+np.ceil(sigmaAvg*2)).reshape((-1,1))),axis=0))[0]
                if len(nearbyPosInd)>(3**2.):
                    temp=np.corrcoef(pos[:,nearbyPosInd])
                    if np.any(np.isnan(temp)):
                        temp=0
                    else:
                        temp=np.linalg.det(temp)
                    corrdet[-1][n]=temp
                elif np.any(pos[-2:,n]<np.ceil(sigmaAvg*2)) or np.any((pos[-2:,n]+np.ceil(sigmaAvg*2))>=image.data.shape[-2:]):
                        corrdet[-1][n]=-1
            if maxCorrdet<corrdet[-1].max():
                maxCorrdet=corrdet[-1].max()
        if maxCorrdet<0.5:
            logger.warning('Warning, determinant or correlation matrix has a maximum value of {0:.3e}'.format(maxCorrdet))
        for t in range(image.data.shape[dimInd]):
            corrdet[t]=corrdet[t]/maxCorrdet
            corrdet[t][corrdet[t]<0]=1.
            newImage.data[(*dimIndSlice,t)][tuple(posALL[t])]*=corrdet[t]
    return newImage
def applyThreshold(image,threshold):
    reduce=int(255/image.data.shape[1])
    newImage=image.clone()
    newImage.data[newImage.data<=(255-threshold*reduce)]=0
    return newImage

def singleOutFilter(speckleImageArray, sigma):
    threshold=speckleImageArray.max()*0.1
    newpoints=np.transpose(np.nonzero(speckleImageArray>threshold))
    val=speckleImageArray[np.nonzero(speckleImageArray>threshold)].copy()
    logger.info('Single out '+repr(len(newpoints))+' number of points')
    for n in range(len(newpoints)-1,-1,-1):
        toRemove=2
        for nn in range(-2,0,-1):
            if newpoints[n][nn]<(2*sigma[nn]) or newpoints[n][nn]>(speckleImageArray.shape[nn]-2*sigma[nn]-1):
                toRemove=0
        for nn in [[1,1],[1,-2],[-2,1],[-2,-2]]:
            if not(toRemove):
                break
            mincoord=newpoints[n][-2:]+np.array(nn)*sigma
            if np.any(np.logical_and(newpoints[:,-2:]>=mincoord,newpoints[:,-2:]<=(mincoord+sigma))):
                toRemove-=1
        if toRemove:
            newpoints=np.delete(newpoints,n,axis=0)
            val=np.delete(val,n,axis=0)
        elif toRemove==1:
            logger.debug('at least i have 1')
    logger.info('    to '+repr(len(newpoints))+' number of points')
    newpoints=np.around(newpoints).astype(int)
    newImageArray=np.zeros(speckleImageArray.shape)
    newImageArray[tuple(newpoints.T)]=val
    return newImageArray

def spreadSpeckle(image,spreadSize,overlay=False,overlayFunc=np.max,averageSigma=True,dim='t'):
    newImg=image.clone()
    percent99=np.percentile(newImg.data[newImg.data>0],99)
    if averageSigma:
        spreadSizeAvg=np.ones(len(spreadSize))*np.mean(spreadSize)
    else:
        spreadSizeAvg=spreadSize
    normVal=(1./gaussian_filter(np.ones(np.ones(len(spreadSizeAvg)).astype(int)), spreadSizeAvg,mode='constant')).max()
    if overlay:
        newImg.removeDim(dim)
        newImg.data=overlayFunc(image.data,axis=image.dim.index(dim))
    newImg.data=normVal*gaussian_filter(newImg.data, [*np.zeros(len(newImg.data.shape)-len(spreadSizeAvg)).astype(int),*spreadSizeAvg],mode='wrap')
    newImg.data*=255/percent99
    newImg.data=np.minimum(255,newImg.data)
    return newImg

def speckleTransform(speckleImageArray,transformFolder,fromTime,toTime=None,totalTimeSteps=None,Eulerian=True,highErrorDim=3):
    fromTime=int(fromTime)
    if type(toTime)!=type(None):
        toTime=int(toTime)
    pos=np.transpose(np.nonzero(speckleImageArray))[:,::-1]
    val=speckleImageArray[np.nonzero(speckleImageArray)].copy()
    if Eulerian:
        if totalTimeSteps:
            #Forward
            currentTime=fromTime
            if type(toTime)==type(None):
                temp_toTime=fromTime-1
            else:
                temp_toTime=toTime
            if temp_toTime<0:
                temp_toTime=totalTimeSteps-1
            posF=[pos.copy()]
            incr=1
            Fcount=0
            while currentTime!=temp_toTime:
                if currentTime+incr<totalTimeSteps:
                    nextTime=currentTime+incr
                else:
                    nextTime=0
                file=transformFolder+'/tstep'+str(currentTime)+'to'+str(nextTime)+'_0.txt'
                if not(os.path.isfile(file)):
                    logger.error('ERROR '+file+' does not exist')
                posF.append(processFunc.transform_img2img(posF[-1],file,savePath=transformFolder))
                currentTime=nextTime
                Fcount+=1
            #Backward
            currentTime=fromTime
            if type(toTime)==type(None):
                temp_toTime=fromTime+1
            else:
                temp_toTime=toTime
            if temp_toTime>=totalTimeSteps:
                temp_toTime=0
            posB=[pos.copy()]
            incr=-1
            Bcount=0
            while currentTime!=temp_toTime:
                if currentTime+incr>=0:
                    nextTime=currentTime+incr
                else:
                    nextTime=totalTimeSteps-1
                file=transformFolder+'/tstep'+str(currentTime)+'to'+str(nextTime)+'_0.txt'
                if not(os.path.isfile(file)):
                    logger.error('ERROR '+file+' does not exist')
                posB.append(processFunc.transform_img2img(posB[-1],file,savePath=transformFolder))
                currentTime=nextTime
                Bcount+=1
            if type(toTime)==type(None):
                Fratio=1./(1.+np.arange(totalTimeSteps)/np.arange(totalTimeSteps,0,-1))
                posF=np.roll(np.array(posF),fromTime,axis=0)
                Fratio=np.roll(Fratio,fromTime)
                posB=np.roll(np.array(posB)[::-1],fromTime+1,axis=0)
                newpos=Fratio.reshape((-1,1,1))*posF+(1-Fratio.reshape((-1,1,1)))*posB
            else:
                newpos=np.array([(Bcount*posF[-1]+Fcount*posB[-1])/(Fcount+Bcount)])
        else:
            currentTime=fromTime
            newpos=pos.copy()
            if currentTime>toTime:
                incr=-1
            elif currentTime<toTime:
                incr=1
            while currentTime!=toTime:
                file=transformFolder+'/tstep'+str(currentTime)+'to'+str(currentTime+incr)+'_0.txt'
                if not(os.path.isfile(file)):
                    logger.error('ERROR '+file+' does not exist')
                newpos=processFunc.transform_img2img(newpos,file,savePath=transformFolder)
                currentTime+=incr
            newpos=np.array([newpos])
    else:
        file=transformFolder+'/tstep'+str(fromTime)+'to'+str(toTime)+'_0.txt'
        if not(os.path.isfile(file)):
            logger.error('ERROR '+file+' does not exist')
        newpos=np.array([processFunc.transform_img2img(pos,file,savePath=transformFolder)])
    if type(toTime)==type(None) and Eulerian and totalTimeSteps:
        runN=totalTimeSteps
    else:
        runN=1
    newArray=[]
    newpos=np.around(newpos).astype(int)
    if highErrorDim and runN>1:
        error=Fratio*(1-Fratio)*totalTimeSteps
        if highErrorDim==True or highErrorDim==0:
            accScaling=1./np.maximum(1.,error)
        elif isinstance(highErrorDim,int):
            if highErrorDim<int((len(error)+1)/2):
                accScaling=1./np.maximum(1.,error/error[np.argmin(error)-highErrorDim])
            else:
                accScaling=np.ones(runN)
        else:
            accScaling=highErrorDim(error)
    else:
        accScaling=np.ones(runN)
    for n in range(runN):
        newArray.append(speckleImageArray.copy())
        get=np.all(np.logical_and(newpos[n]>=0,newpos[n]<np.array(speckleImageArray.shape)[::-1]),axis=-1)
        temppos=newpos[n][get]
        tempval=val[get]
        newArray[-1][:]=0
        newArray[-1][tuple(temppos[:,::-1].T)]=tempval.copy()*accScaling[n]
    if runN==1:
        newArray=newArray[0]
    return np.array(newArray)

