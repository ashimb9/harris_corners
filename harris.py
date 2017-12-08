import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from sys import exit


def harris(img,w = 5,thres = 0,killzone=21,count=0):
    '''
    Evaluates Harris-Stephens corner points for a give RGB or grayscale image
    
    img : array-like, 
    This is the image whose corners are to be located
    
    w : int, optional (default = 5)
    This is the length (=width) of the patch that will be convolved with gradient product matrices
    
    thres : int, optional (default = 0)
    It is the threshold value (i.e. Harris value) for a potential corner point
    
    killzone : int, optional (default = 21)
    Specifices the window length(=width) to be used during non-local-maxima suppression
    
    count : int, optional (default = 0)
    This is the number of corner points to be returned (might be lower if #positive Harris-valued points < count)
    
    Returns
    -------
    A tuple whose elements are the row and column indices of the Harris-Stephens corner
    '''

    if(w%2==0 or killzone%2==0):
        exit("Error: Please enter an odd value for window size and/or killzone.")

    if(len(img.shape)==3):
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Generate gradient and gradient square matrices
    ksize = 5   #Gaussian kernel size for derivatives
    # image ddepth is set to 64F since we dont want negative derivates to be truncated to zero
    imgdx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    imgdy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)

    imgdx2 = np.square(imgdx)
    imgdy2 = np.square(imgdy)
    imgdxdy = np.multiply(imgdx,imgdy)

    #Create Gaussian Filter
    win = np.zeros((w,w))
    center = int((w-1)/2)
    win[center,center] = 1
    win = cv2.GaussianBlur(win, (w, w), 2)    #Converts 'win' into a w-by-w Gaussian filter (sd=2 based on Szeliski, 2011, p190)

    #Convolve window (patch) with each of imgdx2,imgdy2, and imgdxdy
    #Calculate the Harris value matrix R

    alpha = 0.06    #Emprically, the "best value" is apparently between 0.04-0.06
    mx2 = cv2.filter2D(imgdx2,-1,win,borderType=cv2.BORDER_CONSTANT)    #Convolve (actually correlates but we have symmetric filter) the gradient product matrices with the Gaussian window
    my2 = cv2.filter2D(imgdy2, -1, win, borderType=cv2.BORDER_CONSTANT)
    mxy = cv2.filter2D(imgdxdy, -1, win, borderType=cv2.BORDER_CONSTANT)
    detM = (mx2*my2)-(mxy*mxy)
    traceM = mx2+my2
    R = detM - alpha*(np.square(traceM))

    #Thresholding and non-local-maxima suppression
    R[np.where(R<thres)]=0 #threshold appears to be largely arbitrary
    localMax = maximum_filter(R,size=killzone,mode='constant',cval=0)
    R = np.where(localMax == R, R, 0)   #Non-maximal suppression

    #Clear the border areas since they would have been compared to extrapolated zeros
    center = int((killzone - 1) / 2)
    R[:center,:] = 0
    R[-center:,:] = 0
    R[:, :center] = 0
    R[:, -center:] = 0

    # Return the largest N(=count) corners if count>0
    if (count > 0 and count<len(np.where(R>0)[0])):
        highIndex = np.argpartition(R, -count, axis=None)[-count:]
        rowI, colI = np.floor_divide(highIndex, R.shape[1]), np.remainder(highIndex, R.shape[1])
        return (rowI,colI)

    return np.where(R>0)
