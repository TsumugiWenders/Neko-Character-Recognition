import os

import cv2.cv2 as cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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


def read_img(imgp: str) -> np.array:
    _img = cv2.imread(imgp)
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    _img = to_square(_img)
    _img = cv2.resize(_img, SHAPE)
    _img = _img.astype(np.float32)
    _img /= 255
    return np.array(_img)


EMPLOY_PARAM = {
    'model_path': './Project/model-resnet.h5',
    'label_path': './Project/model-tag.txt',
}
MODEL = keras.models.load_model(EMPLOY_PARAM['model_path'])
SHAPE = MODEL.input_shape[1:-1]
img_path = 'DataSet/Test/F-2.jpg'
img = read_img(img_path)
input_image = tf.expand_dims(img, 0)
Preds = MODEL.predict(input_image)
for index in range(0, len(MODEL.layers)):
    layer = MODEL.get_layer(index=index)
    # print(str(index) + "  " + str(layer))
layer_outputs = MODEL.get_layer(name="activation_9").output
activation_model = tf.keras.Model(inputs=MODEL.input, outputs=layer_outputs)
activations = activation_model.predict(input_image)
print(activations)
plt.figure(figsize=(7, 3))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('The original image')
plt.subplot(1, 2, 2)
# plt.matshow(activations[0, :, :, 1],fignum=False)
plt.imshow(activations[0, :, :, 1])
plt.title('Feature Map')
plt.show()
