import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import signal


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


#removes noise from im by blurring
#im is a gray scale picture
#method is the method of noise removal, either uniform or Gaussian
def blurring(im, method):
    image = mpimg.imread(im)
    image = np.array(image, dtype = 'float')
    Img2D = image[:,:,0]*0.21+image[:,:,1]*0.72+image[:,:,2]*0.07 

    if method == 'uniform':
        blur = np.array([[1.0/9.0]*5]*5) #blurring array
        #weighted average of RGB pixels to convert to BW
        image = signal.convolve2d(Img2D, blur)
    
    elif method == 'gaussian':
        k = 5
        filter=np.array([[0]*k]*k,dtype='float')
        for x in range(k):
            for y in range(k):
                filter[x,y]=np.exp(-((x-(k-1)*0.5)**2+(y-(k-1)*0.5)**2)/2.0)
            filter_sum=np.sum(filter)
            filter=filter/filter_sum
        image = signal.convolve2d(Img2D, filter)
        
    return image
        

myIm = blurring('kitty-cat.jpg', 'uniform')
plt.imshow(myIm)




# =============================================================================
# def salt_pepper(im,ps=.01,pp=.1):
#     im1=im[:,:,0].copy()
#     n,m=im1.shape
#     for i in range(n):
#         for j in range(m):
#             b=np.random.uniform()
#             if b<ps:
#                 im1[i,j]=1
#             elif b>1-pp:
#                 im1[i,j]=0
#     noisy_im=[[[im1[i,j]]*3 for j in range(m)] for i in range(n)]
#     return noisy_im
# 
# 
# img=mpimg.imread('kitty-cat.jpg')
# img = img/255.0 # Convert to 64-bit floating point.
# plt.imshow(img)
# plt.show()
# 
# greyImg = img.copy()
# 
# # it is much faster when you vectorize the operations
# greyImg = img[:,:,0]*0.21+img[:,:,1]*0.72+img[:,:,2]*0.07
# # Here we insert a new 3rd dimension, which we then repeat the array
# greyImg = np.repeat(greyImg[:, :, np.newaxis], 3, axis=2)
# greyImg = salt_pepper(greyImg)
# 
# plt.imshow(greyImg, cmap='gray')
# img = greyImg
# 
# plt.show()
# 
# =============================================================================
