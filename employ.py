import cv2.cv2 as cv2
import numpy as np
import os
#
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras as keras
#
from Utils import processing, label_code


EMPLOY_PARAM = {
    'model_path': './Project/model-lt-resnet.h5',
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
    result = MODEL.predict(data,verbose=1)
    print(result)
    result_decode = list()
    for index1 in range(result.shape[0]):
        value = label_code.decode_label(result[index1], LABEL)
        result_decode.append(value)

    return result_decode


def read_file_name(fn: str) -> str:
    fnsz = fn.split("/")[-1].split(".")[0].split("-")[0]
    if fnsz == 'F':
        return 'Fumizuki'
    elif fnsz == 'Y':
        return 'Yayoi'
    elif fnsz == 'U':
        return 'Uzuki'
    elif fnsz == 'M':
        return 'Mikazuki'


#################### 应用 ####################
if __name__ == '__main__':
    # 输入图像地址
    img_repo = []
    for filename in os.listdir('./DataSet/break'):
        img_path = './DataSet/break/' + filename
        img_repo.append(img_path)
    img_data = load_image(img_repo)
    '''
    img_data = load_image(['./DataSet/break/F-1.jpg',
                           './DataSet/break/F-2.jpg',
                           './DataSet/break/F-3.jpg',
                           './DataSet/break/F-4.jpg',
                           './DataSet/break/F-5.jpg',
                           './DataSet/break/F-6.jpg',
                           './DataSet/break/F-7.jpg',
                           './DataSet/break/F-8.jpg'])
    '''
    # 预测结果
    res = predict(img_data)
    #print(res)
    count = 0
    for i in range(0, 32):
        if res[i] == read_file_name(img_repo[i]):
            print("True")
            count = count + 1
    print("正确：" + str(count))
    print("正确率：" + str(count/32.0))

