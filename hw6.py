import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import signal

cat=mpimg.imread('kitty-cat.jpg')

cat = np.array(cat, dtype='float') #array of the cat image


#outputs a pink heart around im
#im is an image
def heart(im):
    image = np.array(mpimg.imread(im), dtype='int')
    height, width, RGB = image.shape #height, weight, color dimension of image

    y,x = np.ogrid[0:height, 0:width] # grid for mask
    left_tri = y >= height - width/2 + x
    right_tri = y >= height + width/2 - x
    left_circ = ((width/4)**2 <= (x - width/4)**2 + \
                 (y - height + width/2)**2) & \
                 (y <= height - width/2) & (x <= width/2)
    right_circ = ((width/4)**2 <= (x - 3*width/4)**2 + \
                  (y - height + width/2)**2) & \
                  (y <= height - width/2) & (x >= width/2)

    image[left_tri] = [255,153,204] # apply masks
    image[right_tri] = [255,153,204]
    image[left_circ] = [255,153,204]
    image[right_circ] = [255,153,204]
    plt.imshow(image)
    
#heart('kitty-cat.jpg')
    


def salt_pepper(im,ps=.01,pp=.1):
    im1=im[:,:,0].copy()
    n,m=im1.shape
    for i in range(n):
        for j in range(m):
            b=np.random.uniform()
            if b<ps:
                im1[i,j]=1
            elif b>1-pp:
                im1[i,j]=0
    noisy_im=[[[im1[i,j]]*3 for j in range(m)] for i in range(n)]
    return noisy_im

#img=mpimg.imread('kitty-cat.jpg')
#img = img/255.0 # Convert to 64-bit floating point.
#plt.imshow(img)
#plt.show()
#
#greyImg = img.copy()
#
## it is much faster when you vectorize the operations
#greyImg = img[:,:,0]*0.21+img[:,:,1]*0.72+img[:,:,2]*0.07
## Here we insert a new 3rd dimension, which we then repeat the array
#greyImg = np.repeat(greyImg[:, :, np.newaxis], 3, axis=2)
#greyImg = salt_pepper(greyImg)
#
#plt.imshow(greyImg, cmap='gray')
#
#plt.show()
#blur = np.array([[1.0/9.0]*3]*3)
#
#Img2D = img[:,:,0]*0.21+img[:,:,1]*0.72+img[:,:,2]*0.07
#img = signal.convolve(Img2D, blur)
#
#
#plt.show()


def average(pixel):
    return (pixel[0]*0.21 + pixel[1]*0.72 + pixel[2]*0.07)


#removes noise from im
#im is a gray scale picture
#method is the method of noise removal, either uniform or Gaussian
def blurring(im, method):
    if method == 'uniform':
        image = mpimg.imread(im)
        image = np.array(image, dtype = 'float')
        
        blur = np.array([[1.0/9.0]*5]*5)
        Img2D = image[:,:,0]*0.21+image[:,:,1]*0.72+image[:,:,2]*0.07
        print Img2D

        image = signal.convolve2d(Img2D, blur)
        
        height, width = image.shape #height, weight, color dimension of image     
    
#        for row in range(height): # every row in image
#           print row
#           if row >=3 and row <= height - 3: # not an edge row
#               for pixel in range(width):
#                   #print pixel
#                   #pixel = average(pixel)
#                   #blur[row][pixel] = image[row-1:row+2,pixel-1:pixel+2]
#       return blur
    elif method == 'gaussian':
        pass

myIm = blurring('kitty-cat.jpg', 'uniform')
plt.imshow(myIm)
        
    
myArray = np.array([[1.0/9.0]*3]*3)
print myArray
