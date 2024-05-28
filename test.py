import random

import cv2.cv2 as cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras as keras
#
from Utils import processing, label_code

EMPLOY_PARAM = {
    'model_path': './Project/model-se-resnet.h5',
    'label_path': './Project/model-tag.txt',
}

MODEL = keras.models.load_model(EMPLOY_PARAM['model_path'])
SHAPE = MODEL.input_shape[1:-1]
LABEL = label_code.load_label(EMPLOY_PARAM['label_path'])


def load_image(paths: list) -> np.ndarray:
    _imgs = list()

    for path in paths:
        _img = cv2.imread(path)
        assert _img is not None

        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _img = processing.to_square(_img)

        _img = cv2.resize(_img, SHAPE)
        _img = _img.astype(np.float32)
        _img /= 255
        _imgs.append(_img)

    return np.array(_imgs)


def predict(data: np.ndarray) -> list:
    result = MODEL.predict(data, verbose=1)
    print(result)
    # print(np.argmax(result[0]),)
    result_decode = list()
    for index1 in range(result.shape[0]):
        value = label_code.decode_label(result[index1], LABEL)
        result_decode.append(value)

    return result_decode


def read_img(imgp: str) -> np.array:
    img = cv2.imread(imgp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_file_name(fn: str) -> str:
    fnsz = fn.split("/")[-1].split(".")[0].split("-")[0]
    if fnsz == 'F':
        return 'True:Fumizuki'
    elif fnsz == 'Y':
        return 'True:Yayoi'
    elif fnsz == 'U':
        return 'True:Uzuki'
    elif fnsz == 'M':
        return 'True:Mikazuki'


def get_img() -> list:
    img_Repo = []
    for filename in os.listdir('./DataSet/Test'):
        img_path = './DataSet/Test/' + filename
        img_Repo.append(img_path)
    FL = []
    YL = []
    ML = []
    UL = []
    for file in img_Repo:
        if 'F' in file:
            FL.append(file)
    for file in img_Repo:
        if 'Y' in file:
            YL.append(file)
    for file in img_Repo:
        if 'U' in file:
            UL.append(file)
    for file in img_Repo:
        if 'M' in file:
            ML.append(file)

    employ_image = []
    while True:
        x = random.choice(FL)
        if x in employ_image:
            continue
        else:
            employ_image.append(x)
        if len(employ_image) == 2:
            break
    while True:
        x = random.choice(ML)
        if x in employ_image:
            continue
        else:
            employ_image.append(x)
        if len(employ_image) == 4:
            break
    while True:
        x = random.choice(YL)
        if x in employ_image:
            continue
        else:
            employ_image.append(x)
        if len(employ_image) == 6:
            break
    while True:
        x = random.choice(UL)
        if x in employ_image:
            continue
        else:
            employ_image.append(x)
        if len(employ_image) == 8:
            break
    return employ_image


def get_Random_img() -> list:
    img_Repo = []
    for filename in os.listdir('./DataSet/Test'):
        img_path = './DataSet/Test/' + filename
        img_Repo.append(img_path)
    employ_image = []
    while True:
        x = random.choice(img_Repo)
        if x in employ_image:
            continue
        else:
            employ_image.append(x)
        if len(employ_image) == 8:
            break
    return employ_image


#################### 应用 ####################
if __name__ == '__main__':
    employ_image = get_img()
    img_data = load_image(employ_image)
    # 预测结果
    res = predict(img_data)
    print(res)
    #绘图
    plt.figure(figsize=(12, 8), dpi=200)
    plt.subplots_adjust(hspace=0.4,wspace=0.2)

    plt.subplot(2, 4, 1)
    plt.imshow(read_img(employ_image[0]))
    plt.title(read_file_name(employ_image[0]))
    plt.xlabel(res[0])
    #plt.ylabel(read_file_name(employ_image[0]))

    plt.subplot(2, 4, 2)
    plt.imshow(read_img(employ_image[1]))
    plt.title(read_file_name(employ_image[1]))
    plt.xlabel(res[1])
    #plt.ylabel(read_file_name(employ_image[1]))

    plt.subplot(2, 4, 3)
    plt.imshow(read_img(employ_image[2]))
    plt.title(read_file_name(employ_image[2]))
    plt.xlabel(res[2])
    #plt.ylabel(read_file_name(employ_image[2]))

    plt.subplot(2, 4, 4)
    plt.imshow(read_img(employ_image[3]))
    plt.title(read_file_name(employ_image[3]))
    plt.xlabel(res[3])
    #plt.ylabel(read_file_name(employ_image[3]))

    plt.subplot(2, 4, 5)
    plt.imshow(read_img(employ_image[4]))
    plt.title(read_file_name(employ_image[4]))
    plt.xlabel(res[4])
    # plt.ylabel(read_file_name(employ_image[0]))

    plt.subplot(2, 4, 6)
    plt.imshow(read_img(employ_image[5]))
    plt.title(read_file_name(employ_image[5]))
    plt.xlabel(res[5])
    # plt.ylabel(read_file_name(employ_image[1]))

    plt.subplot(2, 4, 7)
    plt.imshow(read_img(employ_image[6]))
    plt.title(read_file_name(employ_image[6]))
    plt.xlabel(res[6])
    # plt.ylabel(read_file_name(employ_image[2]))

    plt.subplot(2, 4, 8)
    plt.imshow(read_img(employ_image[7]))
    plt.title(read_file_name(employ_image[7]))
    plt.xlabel(res[7])
    # plt.ylabel(read_file_name(employ_image[3]))

    plt.show()
