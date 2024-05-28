import cv2.cv2 as cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from Utils import processing

test_img = cv2.imread('72737588_p0.jpg')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = processing.to_square(test_img)
test_img = cv2.resize(test_img,(250,250))
a1 = cv2.flip(test_img, random.randint(-1, 1))

xy = int(250 * 0.04)
M = np.array([[1, 0, random.randint(-xy, xy)],[0, 1, random.randint(-xy, xy)]], dtype=np.float32)
a2 = cv2.warpAffine(test_img, M, (250,250), borderValue=(127, 127, 127))


rg = random.randint(5, 5)
M = cv2.getRotationMatrix2D((45, 45), rg, 1)
a3 = cv2.warpAffine(test_img, M, (250,250), borderValue=(127, 127, 127))

plt.figure(figsize=(12, 3))
plt.subplot(1, 4, 1)
plt.imshow(test_img)
plt.title('The original image')
plt.subplot(1, 4, 2)
plt.imshow(a1)
plt.title('Randomly flip the image')
plt.subplot(1, 4, 3)
plt.imshow(a2)
plt.title('Randomly offset images')
plt.subplot(1, 4, 4)
plt.imshow(a3)
plt.title('Rotate the image randomly')
plt.show()
