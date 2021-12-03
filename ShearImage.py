import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test_data.jpg')

rows,cols,ch = img.shape
pts1 = np.float32([[ 232 , 93 ],[ 499 , 199 ],[ 5 , 366 ],[ 313 , 552 ]])
pts2 = np.float32([[ 0 , 0 ],[ 512 , 0 ],[ 0 , 682 ],[ 512 , 682 ]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,( img.shape[1] , img.shape[0] ))

cv2.imwrite('output.jpg', dst)

plt.subplot( 121 ),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title( 'Input' )
plt.subplot( 122 ),plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)),plt.title( 'Output' )

plt.show()