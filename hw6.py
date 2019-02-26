import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

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

    image[left_tri] = [255,153,204]
    image[right_tri] = [255,153,204]
    image[left_circ] = [255,153,204]
    image[right_circ] = [255,153,204]
    plt.imshow(image)
    
#heart('kitty-cat.jpg')
    
#removes noise from im
#im is a gray scale picture
#method is the method of noise removal, either uniform or Gaussian
def blurring(im, method):
    image = np.array(mpimg.imread(im), dtype='int')
    print image
    
blurring('kitty-cat.jpg', "poo")