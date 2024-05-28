import time

import cv2.cv2 as cv2
import os
import random
import numpy as np
import threading
import queue
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img

#
from Utils import processing


# 图像预处理生成器
class Preprocessing:
    # 图片二次缩放概率
    secondary_zoom = 0.3
    # 图片翻转概率
    flip_probability = 0.3
    # 图片偏移范围比率
    offset_range = 0.07
    # 图片旋转范围角度
    rotation_range = 5

    # 关闭子线程
    is_close = False

    def __init__(self, path: str, size: tuple, batch_size: int):
        assert os.path.isdir(path)
        assert len(size) == 2
        assert type(batch_size) == int

        self.path = path
        self.size = size
        self.batch_size = batch_size

        ########## 获取 标签 和 标签路径
        class_path = list()
        class_label = list()

        root_depth = len(path.split(os.path.sep))
        for dirpath, dirnames, filenames in os.walk(path):
            for dirname in dirnames:
                subpath = os.path.join(dirpath, dirname)
                dir_depth = len(subpath.split(os.path.sep))

                if dir_depth == root_depth + 1:
                    class_path.append(subpath)
                    class_label.append(dirname)

        self.class_path = class_path
        self.class_label = class_label
        ########## end

        print('label count:', len(self.class_label))
        print('label:', self.class_label)

    def OneHot_encode(self, label: str) -> np.ndarray:
        onehot = np.zeros(len(self.class_label), dtype=np.float32)
        index = self.class_label.index(label)
        onehot[index] = 1

        return onehot

    def OneHot_decode(self, label: np.ndarray) -> str:
        index = label.argmax()
        return self.class_label[index]

    # 图像预处理
    def preprocessing(self, img: np.ndarray, pre: bool) -> np.ndarray:

        # 二次缩放
        if random.random() < self.secondary_zoom and pre:
            img = processing.secondary_zoom(img)

        # 方形
        img = processing.to_square(img)
        img = cv2.resize(img, self.size)

        if pre:
            # 翻转
            if random.random() < self.flip_probability:
                img = cv2.flip(img, random.randint(-1, 1))
            # 偏移
            if random.random() < 0.8:
                xy = int(min(self.size) * self.offset_range)

                M = np.array([[1, 0, random.randint(-xy, xy)],
                              [0, 1, random.randint(-xy, xy)]], dtype=np.float32)

                img = cv2.warpAffine(img, M, dsize=self.size, borderValue=(127, 127, 127))
            # 旋转
            if random.random() < 0.8:
                rg = random.randint(-self.rotation_range, self.rotation_range)

                M = cv2.getRotationMatrix2D((45, 45), rg, 1)

                img = cv2.warpAffine(img, M, dsize=self.size, borderValue=(127, 127, 127))

        img = img.astype(np.float32)
        img /= 255
        return img

    # 获取生成器
    def get_data_generator(self, pre: bool, shuffle=True):
        data_path = list()

        for label_path in self.class_path:
            data_label = label_path.split(os.path.sep)[-1]
            data_label = self.OneHot_encode(data_label)

            for dirpath, dirnames, filenames in os.walk(label_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    data_path.append((filepath, data_label))

        print(f'[Pre={pre}] image: {len(data_path)} batch_size: {self.batch_size} steps_per_epoch: {len(data_path) // self.batch_size}')

        if shuffle:
            random.shuffle(data_path)

        def generator():
            while True:
                yield q.get()

        def sub():
            data_index: int = -1

            while True:
                batch_data_x = []
                batch_data_y = []

                for _ in range(self.batch_size):
                    data_index += 1
                    data_index = 0 if data_index >= len(data_path) else data_index

                    img = cv2.imread(data_path[data_index][0])
                    if img is None:
                        # 无法识别gif文件
                        # print('Image is None')
                        # print('Path:', data_path[data_index][0])
                        # print()
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = self.preprocessing(img, pre)

                    batch_data_x.append(img)
                    batch_data_y.append(data_path[data_index][1])

                batch_data_x = np.array(batch_data_x)
                batch_data_y = np.array(batch_data_y)

                if q.qsize() == 4:
                    if self.is_close:
                        break
                    else:
                        time.sleep(0.1)
                else:
                    q.put((batch_data_x, batch_data_y))

        # 开启子线程并返回生成器
        q = queue.Queue(4)
        threading.Thread(target=sub).start()
        return generator(), len(data_path) // self.batch_size

    # 关闭子线程
    def close_generator(self):
        self.is_close = True

    # 获取标签名
    def get_label_name(self):
        return self.class_label


if __name__ == '__main__':
    p = Preprocessing(path='../DemoDS',
                      size=(250, 250),
                      batch_size=32,
                      )

    pg, steps_per_epoch = p.get_data_generator(pre=False)
    print('steps_per_epoch:', steps_per_epoch)

    for data_batch, labels_batch in pg:
        print(data_batch.shape, data_batch.dtype)
        print(labels_batch.shape, labels_batch.dtype)

        plt.figure(figsize=(6, 4))
        plt.subplot(2, 2, 1)
        plt.imshow(array_to_img(data_batch[0]))
        plt.subplot(2, 2, 2)
        plt.imshow(array_to_img(data_batch[1]))
        plt.subplot(2, 2, 3)
        plt.imshow(array_to_img(data_batch[2]))
        plt.subplot(2, 2, 4)
        plt.imshow(array_to_img(data_batch[3]))
        plt.show()

        break
