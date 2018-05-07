"""
Author: Patrick Ozark
convolution.py
Objectives:
	- Add to an existing Python package to determine direction of motion with a webcam
	- Coding convolution kernels independently
"""
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2

image = img.imread('snake.jpg')
print(image.shape)

def showconvo(image,kernel,title):
	"""
	Displays the original image next to an image
	convoluted with the input kernel
	"""
	img = image
	kernel = kernel
	title = title

	#Create subplots to go in a 2x1 grid
	plt.subplot(211),plt.imshow(img),plt.title('Original')

	dst = cv2.filter2D(img,-1,kernel)
	plt.subplot(212),plt.imshow(dst),plt.title(title)

	plt.show()



showconvo(image, np.asarray([[0,-1,0],[-1,5,-1],[0,-1,0]]), 'Sharpen')
showconvo(image, np.asarray([[0,1,0],[1,-4,1],[0,1,0]]), 'Edge detection')
