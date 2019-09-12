# medImgProc
MedImgProc is a Medical Image Processing module for viewing and editing.

### imread(imageFile,dimension=None,fileFormat='',crop=None,module=''):
read image file of other format

### imwrite(imageClass,filePath,axes=['y','x'],imageFormat='png',dimRange={},fps=3,color=0):
write image file of other format

### stackImage(imageList,newDim):
stack image file list of medImgProc image format as a new dimension

### apply(imageClass,func,axes=['y','x'],dimSlice={},funcArgs=()):#use slice(a,b) for otherDimLoc 
apply function to medImgProc image format

### arrange(imageClass,newDim,arrangeFront=True):
arrange image array according to the new dimension list (newDim)

### stretch(imageClass,stretchDim,scheme=image.DEFAULT_INTERPOLATION_SCHEME):
resample image according to the ratio in stretchDim Dict using scheme.

### save(imageClass,filePath):
save image file as medImgProc image format

### load(filePath):
load image file of medImgProc image format

### loadStack(imageFileFormat,dimension=None,maxskip=0):
load and stack image file of medImgProc image format with string formating in running order.

### loadmat(fileName,arrayName='',dim=[],dimlen={},dtype=None):
load matlab file to medImgProc image format

### show(imageClass):
show image

## Note
Simple Elastix (https://simpleelastix.github.io/) and OpenCV need to be installed manually  to use certain processing functions.

### Installation
If your default python is python3:
pip install medImgProc
