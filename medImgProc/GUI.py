'''
File: __init__.py
Description: load all class for medImgProc
             Contains externally usable class
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         12JAN2018           - Created
Author: w.x.chan@gmail.com           08OCT2018           - v1.4.0
                                                              -added colortoggler
Author: w.x.chan@gmail.com           08OCT2018           - v1.5.2
                                                              -added Intensity scaling
                                                              - lower slider
Requirements:
    numpy.py
    matplotlib.py
    imageio.py

Known Bug:
    HSV color format not supported
All rights reserved.
'''
_version='1.5.2'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
matplotlib.rc('text', usetex=True)
'''
variables
'''
STD_INSTRUCTIONS='Press Enter to save and exit, Esc to exit\n '
'''
Internal Use functions
'''
def getLastTwoDimArray(imageArray,dimLoad,color=0):
    outputArray=np.copy(imageArray)
    if len(outputArray.shape)>(2+color):
        outputArray=getLastTwoDimArray(outputArray[dimLoad[0]],dimLoad[1:],color=color)
    return outputArray
def getFramePts(pts,dimLoad):
    newPts=np.copy(pts)
    for n in range(len(dimLoad)-2):
        if len(newPts)!=0:
            filterFrame=(newPts[:,n]==dimLoad[n])
            newPts=newPts[filterFrame,:]
        else:
            break
    return newPts
def dimToTitle(dimension,showIndex):
    titleOutput=''
    for n in range(len(dimension)):
        titleOutput+=dimension[n]+':'+str(showIndex[n])
        if n==0:
            titleOutput+=r'$\leftrightarrow$ \hspace{1cm} '
        else:
            titleOutput+=r'$\updownarrow$'
    #titleOutput+='\n '+r'$\leftarrow$ $\rightarrow$ \hspace{1cm} $\uparrow$ $\downarrow$'
    
    return titleOutput

'''
Main GUI class
'''
class image2DGUI:
    def __init__(self,imageClass,addInstruct='',disable=[],initPointList=None,showNow=True):
        self.title=None
        self.addInstruct=STD_INSTRUCTIONS
        if addInstruct!='':
            self.addInstruct+=addInstruct+'\n '
        if type(imageClass)==str:
            self.image=medImgProc.imread(imageClass)
        self.image=imageClass.clone()
        self.image.data=np.maximum(0,self.image.data)
        self.color=0
        if 'RGB' in self.image.dim:
            self.image.rearrangeDim('RGB',False)
            self.color=1
        elif 'RGBA' in self.image.dim:
            self.image.rearrangeDim('RGBA',False)
            self.color=1
        if self.color==1:
            self.addInstruct+='press 1,2,3,.. to toggler color channel and 0 to show all.\n '
            self.colorToggler=[]
        self.fig=plt.figure(1)
        self.showIndex=[]
        for n in range(len(self.image.data.shape)-self.color):
            self.showIndex.append(0)
        self.disable=disable
        self.connectionID=[]
        if 'click' not in self.disable:
            self.connectionID.append(self.fig.canvas.mpl_connect('button_press_event', self.onclick))
        self.connectionID.append(self.fig.canvas.mpl_connect('key_press_event',self.onKeypress))   
        self.enter=False
        self.scaleVisual=1.
        self.logVisual=0.
        
        
        '''
        Return parameters
        '''
        if initPointList is None:
            self.points=np.empty((0,len(self.image.dim)))
        else:
            '''sanitize to show'''
            initPointList=np.copy(initPointList)
            for n in range(len(initPointList)):
                for m in range(len(initPointList[n])-2):
                    initPointList[n,m]=np.round(initPointList[n,m])
            self.points=initPointList
        self.axslide=[]
        self.sSlide=[]
        self.loadImageFrame()

        if showNow:
            plt.show()
    def sliderUpdate(self,val):
        for n in range(len(self.showIndex)-2):
            self.showIndex[n]=int(self.sSlide[n].val)
        self.scaleVisual=self.sSlide[-2].val
        self.logVisual=self.sSlide[-1].val
        self.showNewFrame()
    def onclick(self,event):
        if not(event.dblclick) and event.button==1 and event.inaxes==self.ax:
            newPt=np.array([*self.showIndex[:-2],event.ydata,event.xdata])
            self.points=np.vstack((self.points,newPt))
        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
        self.showNewPoints()
    def onKeypress(self,event):
        if event.key == 'escape':#return values and quit
            for connection in self.connectionID:
                self.fig.canvas.mpl_disconnect(connection)
            plt.close(event.canvas.figure)
        elif event.key == 'enter':
            self.enter=True
            for connection in self.connectionID:
                self.fig.canvas.mpl_disconnect(connection)
            plt.close(event.canvas.figure)
        elif event.key=='up':
            self.switchFrame(-3,1)
        elif event.key=='down':
            self.switchFrame(-3,-1)
        elif event.key=='right':
            self.switchFrame(-4,1)
        elif event.key=='left':
            self.switchFrame(-4,-1)
        elif event.key=='ctrl+z':
            self.removeLastPoint()
        elif self.color==1 and event.key=='0':
            self.colorToggler=[]
            self.showNewFrame()
        elif self.color==1 and event.key in ['1','2','3','4','5','6','7','8','9']:
            if (int(event.key)-1) in self.colorToggler:
                self.colorToggler.remove(int(event.key)-1)
            elif self.image.data.shape[-1] > (int(event.key)-1):
                self.colorToggler+=[int(event.key)-1]
            self.showNewFrame()
        elif event.key in self.image.dim:
            self.swapFrame(event.key)
        else:
            print(event.key)

    def switchFrame(self,index,val=1):
        if len(self.showIndex)>=(-index):
            self.showIndex[index]+=val
            if self.showIndex[index]>(self.image.data.shape[index]-1):
                self.showIndex[index]=0
            elif self.showIndex[index]<0:
                self.showIndex[index]=self.image.data.shape[index]-1
            self.sSlide[len(self.showIndex)+index].set_val(self.showIndex[index])
        self.showNewFrame()
    def swapFrame(self,axis):
        if 'swap' not in self.disable:
            if self.color:
                transposeIndex=self.image.rearrangeDim([axis,self.image.dim[-1]],arrangeFront=False)
            else:
                transposeIndex=self.image.rearrangeDim([axis],arrangeFront=False)
            newShowIndex=[]
            if self.color:
                transposeIndex=transposeIndex[:-1]
            for n in range(len(transposeIndex)):
                newShowIndex.append(self.showIndex[transposeIndex[n]])
            self.showIndex=newShowIndex
            self.points=self.points[:,transposeIndex]
            plt.clf()
            self.loadImageFrame()
            self.showNewFrame()
    def showNewFrame(self):
        newShowImage=getLastTwoDimArray(self.image.data,self.showIndex,color=self.color)
        if self.color:
            newShowImage[...,tuple(self.colorToggler)]=0
        newShowImage=np.maximum(0,np.minimum(255,(newShowImage*self.scaleVisual)**(10.**self.logVisual))).astype('uint8')
        self.main.set_data(newShowImage)
        self.ax.set_aspect(self.image.dimlen[self.image.dim[-2-self.color]]/self.image.dimlen[self.image.dim[-1-self.color]])
        self.showNewPoints()
        
        
        pp=plt.setp(self.title,text=self.addInstruct+dimToTitle(self.image.dim[:-2-self.color],self.showIndex[:-2]))
        self.fig.canvas.draw()
    def showNewPoints(self):
        showpoints=getFramePts(self.points,self.showIndex)
        self.ptplt.set_offsets(showpoints[:,[-1,-2]])
        self.fig.canvas.draw()
        
    def loadImageFrame(self):
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=(len(self.showIndex)+3)*0.04)
        showImage=getLastTwoDimArray(self.image.data,self.showIndex,color=self.color)
        if self.color:
            showImage[...,tuple(self.colorToggler)]=0
        showImage=np.maximum(0,np.minimum(255,(showImage*self.scaleVisual)**(10.**self.logVisual))).astype('uint8')
        self.main=self.ax.imshow(showImage,cmap=matplotlib.cm.gray, vmin=0, vmax=255)
        self.ax.set_aspect(self.image.dimlen[self.image.dim[-2-self.color]]/self.image.dimlen[self.image.dim[-1-self.color]])

        showpoints=getFramePts(self.points,self.showIndex)
        self.ptplt=self.ax.scatter(showpoints[:,-1],showpoints[:,-2],color='r',marker='x')

        self.title=plt.title(self.addInstruct+dimToTitle(self.image.dim[:-2-self.color],self.showIndex[:-2]))
        plt.ylabel(self.image.dim[-2-self.color])
        plt.xlabel(self.image.dim[-1-self.color])

        '''set slider for image control'''
        axcolor = 'lightgoldenrodyellow'
        self.axslide=[]
        self.sSlide=[]
        for n in range(len(self.showIndex)-2):
            self.axslide.append(self.fig.add_axes([0.1, 0.02+n*0.04, 0.65, 0.03], facecolor=axcolor))
            self.sSlide.append(Slider(self.axslide[-1], self.image.dim[n], 0, self.image.data.shape[n]-1, valinit=self.showIndex[n],valfmt="%i"))
            self.sSlide[-1].on_changed(self.sliderUpdate)
        self.axslide.append(self.fig.add_axes([0.1, 0.02+(len(self.showIndex)-2)*0.04, 0.65, 0.03], facecolor=axcolor))
        self.sSlide.append(Slider(self.axslide[-1], 'Iscale', 0.1, 10., valinit=self.scaleVisual,valstep=0.1))
        self.sSlide[-1].on_changed(self.sliderUpdate)
        self.axslide.append(self.fig.add_axes([0.1, 0.02+(len(self.showIndex)-1)*0.04, 0.65, 0.03], facecolor=axcolor))
        self.sSlide.append(Slider(self.axslide[-1], 'logPOW', -1., 1., valinit=self.logVisual,valstep=0.02))
        self.sSlide[-1].on_changed(self.sliderUpdate)
        
    def removeLastPoint(self):
        self.points=self.points[:-1,:]
        self.showNewPoints()
    def show(self):
        plt.show()
