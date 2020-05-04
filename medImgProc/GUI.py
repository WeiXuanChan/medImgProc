'''
File: GUI.py
Description: load all class for medImgProc
             Contains externally usable class
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         12JAN2019           - Created
Author: w.x.chan@gmail.com           08OCT2019           - v1.4.0
                                                              -added colortoggler
Author: w.x.chan@gmail.com           08OCT2019           - v1.5.2
                                                              -added Intensity scaling
                                                              - lower slider
Author: w.x.chan@gmail.com           10Jan2020           - v2.3.9
                                                              -added cubic spline line drawing
                                                              -removed latex dependency
                                                              -debug function show() in image2DGUI
Author: w.x.chan@gmail.com           12Jan2020           - v2.3.10
                                                              -debug keypress switch frame of rgb image
Author: w.x.chan@gmail.com           23MAR2020           - v2.6.4
                                                              -added color contour 
Author: w.x.chan@gmail.com           24MAR2020           - v2.6.13
                                                              -allow save image
Author: w.x.chan@gmail.com           29APR2020           - v2.6.19
                                                              -allow switch line
Requirements:
    numpy.py
    matplotlib.py
    imageio.py

Known Bug:
    HSV color format not supported
All rights reserved.
'''
_version='2.6.19'
import logging
logger = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from scipy import interpolate
#matplotlib.rc('text', usetex=True)
try:
    import tkinter
    from tkinter import filedialog
    import os
except:
    pass
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
    newPts=np.array(pts)
    for n in range(len(dimLoad)-2):
        if len(newPts)!=0:
            filterFrame=(newPts[:,n]==dimLoad[n])
            newPts=newPts[filterFrame,:]
        else:
            break
    return newPts
def dimToTitle(dimension,showIndex):
    titleOutput=''
    for n in range(-1,-len(dimension)-1,-1):
        titleOutput+=dimension[n]+':'+str(showIndex[n])
        if n==-1:
            titleOutput+='^v   '  #r'$\updownarrow$'
            
        else:
            titleOutput+='<>'#r'$\leftrightarrow$ \hspace{1cm} '
            break
    #titleOutput+='\n '+r'$\leftarrow$ $\rightarrow$ \hspace{1cm} $\uparrow$ $\downarrow$'
    
    return titleOutput

'''
Main GUI class
'''
class image2DGUI:
    def __init__(self,imageClass,addInstruct='',disable=[],initPointList=None,initLineList=None,manualToleranceRatio=0.05,showNow=True,contourImageArray=None):
        self.title=None
        self.addInstruct=STD_INSTRUCTIONS
        if addInstruct!='':
            self.addInstruct+=addInstruct+'\n '
        if type(imageClass)==str:
            self.image=medImgProc.imread(imageClass)
        self.image=imageClass.clone()
        self.image.data=np.maximum(0,self.image.data)
        if contourImageArray is None:
            self.contourImage=None
        else:
            self.contourImage=imageClass.clone()
            self.contourImage.data[:]=contourImageArray
            self.contourImage.data=self.contourImage.data.astype(int)
        self.color=0
        if 'RGB' in self.image.dim:
            self.image.rearrangeDim('RGB',False)
            if self.contourImage is not None:
                self.contourImage.rearrangeDim('RGB',False)
            self.color=1
        elif 'RGBA' in self.image.dim:
            self.image.rearrangeDim('RGBA',False)
            if self.contourImage is not None:
                self.contourImage.rearrangeDim('RGBA',False)
            self.color=1
        if self.color==1:
            self.addInstruct+='press 1,2,3,.. to toggler color channel and 0 to show all.\n '
            self.colorToggler=[]
        self.fig=plt.figure(1)
        self.showIndex=[]
        for n in range(len(self.image.data.shape)-self.color):
            self.showIndex.append(0)
        self.disable=disable
        self.sliderLoc=0.02
        self.connectionID=[]
        if 'click' not in self.disable or 'line' not in self.disable:
            self.connectionID.append(self.fig.canvas.mpl_connect('button_press_event', self.onclick))
        if 'line' not in self.disable:
            self.sliderLoc+=0.06
            self.connectionID.append(self.fig.canvas.mpl_connect('button_release_event', self.onrelease))
            self.connectionID.append(self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion))
            
        self.connectionID.append(self.fig.canvas.mpl_connect('key_press_event',self.onKeypress))   
        self.enter=False
        self.scaleVisual=1.
        self.logVisual=0.
        self.line_selected=-1
        self.point_selected=-1
        self.show_line=False
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
            
        
        self.lineplt=[]
        self.lines=[]
        if initLineList is not None:
            for line in initLineList:
                self.lines.append(line.copy())
            self.show_line=True
                
        self.axslide=[]
        self.sSlide=[]
        
        self.manualToleranceRatio=manualToleranceRatio
        self.manualTolerance=(max(*self.image.data.shape[-2:])*self.manualToleranceRatio)**2.
        
        self.loadImageFrame()
        if showNow:
            plt.show()
    def sliderUpdate(self,val):
        self.line_selected=-1
        for n in range(len(self.showIndex)-2):
            self.showIndex[n]=int(self.sSlide[n].val)
        self.scaleVisual=self.sSlide[-2].val
        self.logVisual=self.sSlide[-1].val
        self.showNewFrame()
        if self.show_line:
            for nline in range(len(self.lines)):
                self.showNewLine(nline)
    def getLineIndex(self,y,x):
        chosen=None
        distance=float('inf')
        if self.line_selected<0:
            for nline in range(len(self.lines)):
                points=getFramePts(self.lines[nline],self.showIndex)
                if len(points)>1:
                    tck,temp = interpolate.splprep([points[:,-1], points[:,-2]], s=0,k=min(4,len(points))-1)
                    cspline_detectline = np.array(interpolate.splev(np.arange(0, 1.01, 0.01), tck)).T
                    distance_temp=np.min(np.sum((cspline_detectline-np.array([[x,y]]))**2.,axis=1))
                    if distance_temp<distance:
                        distance=distance_temp
                        chosen=nline
                elif len(points)==1:
                    distance_temp=np.min(np.sum((np.array([points[:,-1], points[:,-2]])-np.array([[x,y]]))**2.,axis=1))
                    if distance_temp<distance:
                        distance=distance_temp
                        chosen=nline
        elif len(self.lines[self.line_selected])>0:
            points=getFramePts(self.lines[self.line_selected],self.showIndex)
            distance=np.sum((points[:,-2:]-np.array([[y,x]]))**2.,axis=1)
            chosen=np.argmin(distance)
        return (chosen,distance)
    def onclick(self,event):
        if not(event.dblclick) and event.button==1 and event.inaxes==self.ax:
            newPt=np.array([*self.showIndex[:-2],event.ydata,event.xdata])
            if 'click' not in self.disable:
                self.points=np.vstack((self.points,newPt))
                self.showNewPoints()
            if self.show_line:
                lineIndex,distance=self.getLineIndex(event.ydata,event.xdata)
                if self.line_selected>=0:
                    addlinept=len(self.lines[self.line_selected])
                    if lineIndex is not None:
                        if distance.min() < self.manualTolerance:
                            self.point_selected=lineIndex
                            addlinept=None
                        elif addlinept>1:
                            detectline=getFramePts(self.lines[self.line_selected],self.showIndex)
                            tck,temp = interpolate.splprep([detectline[:,-1], detectline[:,-2]], s=0,k=min(4,len(detectline))-1)
                            cspline_detectline = np.array(interpolate.splev(np.arange(0, 1.+1./100./(addlinept-1), 1./100./(addlinept-1)), tck)).T
                            detectdistance=np.sum((cspline_detectline-np.array([[event.xdata,event.ydata]]))**2.,axis=1)
                            if detectdistance.min()<self.manualTolerance:
                                splineIndex=detectdistance.argmin()
                                for npoint in range(1,addlinept):
                                    if splineIndex<np.argmin(np.sum((cspline_detectline-np.array([[self.lines[self.line_selected][npoint][-1],self.lines[self.line_selected][npoint][-2]]]))**2.,axis=1)):
                                        addlinept=npoint
                                        break
                    if addlinept is not None:
                        self.point_selected=-1
                        self.lines[self.line_selected].insert(addlinept,newPt)
                    self.showNewLine(self.line_selected)
                elif lineIndex is not None:
                    if self.line_selected==-1:
                        self.line_selected=lineIndex
                        self.showNewLine(lineIndex)
                    elif self.line_selected==-2:
                        self.lines.pop(lineIndex)
                        self.line_selected=-1
                        self.loadImageFrame()
                    
    def onrelease(self,event):
        self.point_selected=-1
        if self.line_selected>=0:
            if len(self.lines[self.line_selected])>0:
                self.showNewLine(self.line_selected)
        return
    def onmotion(self,event):
        if self.line_selected<0 or self.point_selected<0:
            return
        if event.xdata is None or event.ydata is None:
            if self.line_selected>=0 and self.point_selected>=0:
                self.lines[self.line_selected].pop(self.point_selected)
                self.point_selected=-1
                if len(self.lines[self.line_selected])>0:
                   self.showNewLine(self.line_selected)
                else:
                   self.lines.pop(self.line_selected)
                   self.line_selected=-1
                   self.loadImageFrame()
            return
        newPt=np.array([*self.showIndex[:-2],event.ydata,event.xdata])
        self.lines[self.line_selected][self.point_selected]=newPt.copy()
        self.showNewLine(self.line_selected)
        return
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
            logger.info(event.key)
    def save_image(self,event):
        self.ax.axis('off')
        extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        maxdpi=max(self.image.data.shape[-2-self.color]/extent.height,self.image.data.shape[-1-self.color]/extent.width)
        root = tkinter.Tk()
        fileName=root.withdraw()
        fileName=''.join(self.image.dim[(-2-self.color):][:2])
        for dimN in range(len(self.image.dim)-2-self.color):
            fileName+='_'+self.image.dim[dimN]+str(self.showIndex[dimN])
        tempdir = filedialog.asksaveasfilename(parent=root, initialdir=os.getcwd(), title='Save Image '+fileName+' as')
        if isinstance(tempdir,str):
            if tempdir[-4:]!='.png':
                tempdir+='.png'
            self.fig.savefig(tempdir, bbox_inches=extent,dpi=maxdpi)
        root.destroy()
        self.ax.axis('on')
    def togger_line(self,event):
        self.show_line=not(self.show_line)
        if not(self.show_line):
            self.line_selected=-1
        self.loadImageFrame()
    def new_line(self,event):
        self.show_line=True
        if len(self.lineplt)>self.line_selected and self.line_selected>=0:
            if len(self.lineplt[self.line_selected])>1:
                self.lineplt[self.line_selected][1].set_color('b')
        self.line_selected=len(self.lines)
        self.lines.append([])
    def edit_line(self,event):
        if len(self.lineplt)>self.line_selected:
            if len(self.lineplt[self.line_selected])>1:
                self.lineplt[self.line_selected][1].set_color('b')
        self.line_selected=-1
    def switch_line(self,event):
        temp_select=self.line_selected
        if (len(self.lines)-1)<=self.line_selected:
            self.line_selected=-1
            if temp_select!=-1:
                self.showNewLine(temp_select)
        else:
            self.line_selected+=1
            if temp_select!=-1:
                self.showNewLine(temp_select)
                self.showNewLine(self.line_selected)
    def del_line(self,event):
        self.line_selected=-2
    def switchFrame(self,index,val=1):
        self.line_selected=-1
        if len(self.showIndex)>=(-index):
            self.showIndex[index]+=val
            if self.showIndex[index]>(self.image.data.shape[len(self.showIndex)+index]-1):
                self.showIndex[index]=0
            elif self.showIndex[index]<0:
                self.showIndex[index]=self.image.data.shape[len(self.showIndex)+index]-1
            self.sSlide[len(self.showIndex)+index].set_val(self.showIndex[index])
        if self.show_line:
            self.loadImageFrame()
        else:
            self.showNewFrame()
    def swapFrame(self,axis):
        self.line_selected=-1
        if 'swap' not in self.disable:
            if self.color:
                transposeIndex=self.image.rearrangeDim([axis,self.image.dim[-1]],arrangeFront=False)
                if self.contourImage is not None:
                    self.contourImage.rearrangeDim([axis,self.image.dim[-1]],arrangeFront=False)
            else:
                transposeIndex=self.image.rearrangeDim([axis],arrangeFront=False)
                if self.contourImage is not None:
                    self.contourImage.rearrangeDim([axis],arrangeFront=False)
            newShowIndex=[]
            if self.color:
                transposeIndex=transposeIndex[:-1]
            for n in range(len(transposeIndex)):
                newShowIndex.append(self.showIndex[transposeIndex[n]])
            self.showIndex=newShowIndex
            self.points=self.points[:,transposeIndex]
            for nline in range(len(self.lines)):
                self.lines[nline]=self.lines[nline][:,transposeIndex]
            self.manualTolerance=(max(*self.image.data.shape[-2:])*self.manualToleranceRatio)**2.
            self.loadImageFrame()
            self.showNewFrame()
    def showNewFrame(self):
        newShowImage=getLastTwoDimArray(self.image.data,self.showIndex,color=self.color)
        if self.color:
            newShowImage[...,tuple(self.colorToggler)]=0
        newShowImage=np.maximum(0,np.minimum(255,(newShowImage*self.scaleVisual)**(10.**self.logVisual))).astype('uint8')
        self.main.set_data(newShowImage)
        self.ax.set_aspect(self.image.dimlen[self.image.dim[-2-self.color]]/self.image.dimlen[self.image.dim[-1-self.color]])
        if self.contourImage is not None:
            if self.contour is not None:
                for coll in self.contour.collections:
                    coll.remove()
            showContour=getLastTwoDimArray(self.contourImage.data,self.showIndex,color=0)
            getlevels=np.arange(0.5+showContour.min(),showContour.max(),1)
            if len(getlevels)>0:
                self.contour=self.ax.contour(showContour,getlevels,linewidths=1.2)
            else:
                self.contour=None
        self.showNewPoints()
        
        pp=plt.setp(self.title,text=self.addInstruct+dimToTitle(self.image.dim[:-2-self.color],self.showIndex[:-2]))
        self.fig.canvas.draw()
    def showNewPoints(self):
        showpoints=getFramePts(self.points,self.showIndex)
        self.ptplt.set_offsets(showpoints[:,[-1,-2]])
        self.fig.canvas.draw()
    def showNewLine(self,lineIndex):
        temp_color='b'
        if lineIndex==self.line_selected:
            temp_color='r'
        showline=getFramePts(self.lines[lineIndex],self.showIndex)
        if len(showline)<=0:
            self.lineplt[lineIndex][0].set_visible(False)
            self.lineplt[lineIndex][1].set_visible(False)
        else:
            if len(self.lineplt)<=lineIndex:
                self.loadImageFrame()
                return;
            elif type(self.lineplt[lineIndex])!=list:
                self.loadImageFrame()
                return;
            self.lineplt[lineIndex][0].set_offsets(showline[:,[-1,-2]])
            self.lineplt[lineIndex][0].set_visible(True)
            self.lineplt[lineIndex][1].set_visible(True)
            if len(showline)>1:
                self.lineplt[lineIndex][0].set_color('b')
                tck,temp = interpolate.splprep([showline[:,-1], showline[:,-2]], s=0,k=min(4,len(showline))-1)
                cspline_line = interpolate.splev(np.arange(0, 1.01, 0.01), tck)
                self.lineplt[lineIndex][1].set_xdata(cspline_line[0])
                self.lineplt[lineIndex][1].set_ydata(cspline_line[1])
                self.lineplt[lineIndex][1].set_color(temp_color)
                #self.lineplt[lineIndex][1].set_offsets(np.array(cspline_line[0],cspline_line[1]).T)
            else:
                self.lineplt[lineIndex][0].set_color('r')
                self.lineplt[lineIndex][1].set_xdata(np.array([showline[0,-1],showline[0,-1]]))
                self.lineplt[lineIndex][1].set_ydata(np.array([showline[0,-2],showline[0,-2]]))
                self.lineplt[lineIndex][1].set_color(temp_color)
                #self.lineplt[lineIndex][1].set_offsets(np.array([showline[0,[-1,-2]],showline[0,[-1,-2]]]))
            if self.line_selected>=0 and self.point_selected>=0:
                if type(self.lineplt[-1])==list:
                    self.lineplt.append(self.ax.scatter([self.lines[self.line_selected][self.point_selected][-1]],[self.lines[self.line_selected][self.point_selected][-2]],color='r',marker='x'))
                else:
                    self.lineplt[-1].set_offsets([self.lines[self.line_selected][self.point_selected][-1],self.lines[self.line_selected][self.point_selected][-2]])
                    self.lineplt[-1].set_visible(True)
            else:
                if type(self.lineplt[-1])!=list:
                    self.lineplt[-1].set_visible(False)
        self.fig.canvas.draw()
    def loadImageFrame(self):
        plt.clf()
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=(len(self.showIndex)+3)*0.04)
        showImage=getLastTwoDimArray(self.image.data,self.showIndex,color=self.color)
        if self.color:
            showImage[...,tuple(self.colorToggler)]=0
        showImage=np.maximum(0,np.minimum(255,(showImage*self.scaleVisual)**(10.**self.logVisual))).astype('uint8')
        self.main=self.ax.imshow(showImage,cmap=matplotlib.cm.gray, vmin=0, vmax=255)
        self.ax.set_aspect(self.image.dimlen[self.image.dim[-2-self.color]]/self.image.dimlen[self.image.dim[-1-self.color]])
        if self.contourImage is not None:
            showContour=getLastTwoDimArray(self.contourImage.data,self.showIndex,color=0)
            getlevels=np.arange(0.5+showContour.min(),showContour.max(),1)
            if len(getlevels)>0:
                self.contour=self.ax.contour(showContour,getlevels,linewidths=1.2)
            else:
                self.contour=None
        showpoints=getFramePts(self.points,self.showIndex)
        self.ptplt=self.ax.scatter(showpoints[:,-1],showpoints[:,-2],color='r',marker='x')
        if self.show_line:
            self.lineplt=[]
            for nline in range(len(self.lines)):
                temp_color='b'
                if nline==self.line_selected:
                    temp_color='r'
                if len(self.lines[nline])>0:
                    self.lineplt.append([self.ax.scatter(np.array(self.lines[nline])[:,-1],np.array(self.lines[nline])[:,-2],color='b',marker='x')])
                    if len(self.lines[nline])>1:
                        tck,temp = interpolate.splprep([np.array(self.lines[nline])[:,-1], np.array(self.lines[nline])[:,-2]], s=0,k=min(4,len(self.lines[nline]))-1)
                        cspline_line = interpolate.splev(np.arange(0, 1.01, 0.01), tck)
                        self.lineplt[-1]+=self.ax.plot(cspline_line[0].copy(),cspline_line[1].copy(),color=temp_color)
                    else:
                        self.lineplt[-1]+=self.ax.plot([self.lines[nline][0][-1],self.lines[nline][0][-1]],[self.lines[nline][0][-2],self.lines[nline][0][-2]],color=temp_color)
                        self.lineplt[-1][0].set_color('r')
                    if len(self.showIndex)>2:
                        if np.any(self.lines[nline][0][:(len(self.showIndex)-2)]!=self.showIndex[:-2]):
                            self.lineplt[-1][0].set_visible(False)
                            self.lineplt[-1][1].set_visible(False)
            if self.line_selected>=0 and self.point_selected>=0:
                self.lineplt.append(self.ax.scatter([self.lines[self.line_selected][self.point_selected][-1]],[self.lines[self.line_selected][self.point_selected][-2]],color='r',marker='x'))
        self.title=plt.title(self.addInstruct+dimToTitle(self.image.dim[:-2-self.color],self.showIndex[:-2]))
        plt.ylabel(self.image.dim[-2-self.color])
        plt.xlabel(self.image.dim[-1-self.color])

        '''set slider for image control'''
        axcolor = 'lightgoldenrodyellow'
        self.saveImageButton=Button(self.fig.add_axes([0.75, 0.02, 0.1, 0.05]), 'Save')
        self.saveImageButton.on_clicked(self.save_image)
        
        self.lineControl=[]
        if 'line' not in self.disable:
            self.lineControl.append(Button(self.fig.add_axes([0.05, 0.02, 0.1, 0.05]), 'Line'))
            self.lineControl[0].on_clicked(self.togger_line)
            self.lineControl.append(Button(self.fig.add_axes([0.2, 0.02, 0.1, 0.05]), 'new'))
            self.lineControl[1].on_clicked(self.new_line)
            self.lineControl.append(Button(self.fig.add_axes([0.35, 0.02, 0.1, 0.05]), 'edit...'))
            self.lineControl[2].on_clicked(self.edit_line)
            self.lineControl.append(Button(self.fig.add_axes([0.5, 0.02, 0.1, 0.05]), 'del...'))
            self.lineControl[3].on_clicked(self.del_line)
            self.lineControl.append(Button(self.fig.add_axes([0.65, 0.02, 0.1, 0.05]), 'switch'))
            self.lineControl[4].on_clicked(self.switch_line)
            
        self.axslide=[]
        self.sSlide=[]
        for n in range(len(self.showIndex)-2):
            self.axslide.append(self.fig.add_axes([0.1, self.sliderLoc+n*0.04, 0.65, 0.03], facecolor=axcolor))
            self.sSlide.append(Slider(self.axslide[-1], self.image.dim[n], 0, self.image.data.shape[n]-1, valinit=self.showIndex[n],valfmt="%i"))
            self.sSlide[-1].on_changed(self.sliderUpdate)
        self.axslide.append(self.fig.add_axes([0.1, self.sliderLoc+(len(self.showIndex)-2)*0.04, 0.65, 0.03], facecolor=axcolor))
        self.sSlide.append(Slider(self.axslide[-1], 'Iscale', 0.1, 10., valinit=self.scaleVisual,valstep=0.1))
        self.sSlide[-1].on_changed(self.sliderUpdate)
        self.axslide.append(self.fig.add_axes([0.1, self.sliderLoc+(len(self.showIndex)-1)*0.04, 0.65, 0.03], facecolor=axcolor))
        self.sSlide.append(Slider(self.axslide[-1], 'logPOW', -1., 1., valinit=self.logVisual,valstep=0.02))
        self.sSlide[-1].on_changed(self.sliderUpdate)
        
    def removeLastPoint(self):
        self.points=self.points[:-1,:]
        self.showNewPoints()
    def show(self):
        self.fig=plt.figure(1)
        self.sliderLoc=0.02
        self.connectionID=[]
        if 'click' not in self.disable or 'line' not in self.disable:
            self.connectionID.append(self.fig.canvas.mpl_connect('button_press_event', self.onclick))
        if 'line' not in self.disable:
            self.sliderLoc+=0.06
            self.connectionID.append(self.fig.canvas.mpl_connect('button_release_event', self.onrelease))
            self.connectionID.append(self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion))
            
        self.connectionID.append(self.fig.canvas.mpl_connect('key_press_event',self.onKeypress))   
        self.enter=False
        self.loadImageFrame()
        plt.show()
