import cv2
import matplotlib.pyplot as plt

img1 = cv2.cvtColor(cv2.imread('./DataSet/Test/F-7.jpg'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('./DataSet/Test/M-4.jpg'), cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(cv2.imread('./DataSet/Test/Y-7.jpg'), cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(cv2.imread('./DataSet/Test/U-7.jpg'), cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 3))
plt.subplot(1, 4, 1)
plt.imshow(img1)
plt.title('Fumizuki')
plt.subplot(1, 4, 2)
plt.imshow(img2)
plt.title('Mikazuki')
plt.subplot(1, 4, 3)
plt.imshow(img3)
plt.title('Yayoi')
plt.subplot(1, 4, 4)
plt.imshow(img4)
plt.title('Uzuki')
plt.show()

