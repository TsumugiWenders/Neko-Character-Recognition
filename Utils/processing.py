import cv2.cv2 as cv2
import numpy as np
import random
import matplotlib.pyplot as plt


# 二次缩放
def secondary_zoom(img: np.ndarray) -> np.ndarray:
    orig_x = img.shape[0]
    orig_y = img.shape[1]

    # 半像素
    HP_x = img.shape[0] // 2
    HP_y = img.shape[1] // 2

    # 缩小使用 区域插值
    img = cv2.resize(img, (HP_y, HP_x), interpolation=cv2.INTER_AREA)

    # 放大
    if random.random() < 0.5:
        # 双线性插值
        img = cv2.resize(img, (orig_y, orig_x), interpolation=cv2.INTER_LINEAR)
    else:
        # 最近邻插值
        img = cv2.resize(img, (orig_y, orig_x), interpolation=cv2.INTER_NEAREST)

    return img


# 转为方形
def to_square(img: np.ndarray) -> np.ndarray:
    if img.shape[0] == img.shape[1]:
        return img

    square_size = max(img.shape)
    square_img = np.ones((square_size, square_size, 3), dtype=np.uint8) * 127

    img_x = img.shape[0]
    img_y = img.shape[1]

    index_x1 = (square_size - img_x) // 2
    index_x2 = index_x1 + img_x
    index_y1 = (square_size - img_y) // 2
    index_y2 = index_y1 + img_y

    square_img[index_x1:index_x2, index_y1:index_y2, :] = img

    return square_img


if __name__ == '__main__':
    test_img = cv2.imread('72737588_p0.jpg')
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    res_img = secondary_zoom(test_img)
    res_img = to_square(res_img)


    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread('72737588_p0.jpg'))
    plt.title('openCV reads in raw image')
    plt.subplot(1, 3, 2)
    plt.imshow(test_img)
    plt.title('Image after gamut conversion')
    plt.subplot(1, 3, 3)
    plt.imshow(res_img)
    plt.title('result image')
    plt.show()
