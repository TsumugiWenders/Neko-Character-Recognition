import keras
from keras.utils.vis_utils import plot_model

EMPLOY_PARAM = {
    'model_path': './Project/model-se-resnet.h5',
    'label_path': './Project/model-tag.txt',
}
MODEL = keras.models.load_model(EMPLOY_PARAM['model_path'])
MODEL.summary()


plot_model(MODEL,
           to_file='./Project/model-structure.png',
           show_shapes=True,
           show_dtype=True,
           show_layer_names=True,
           dpi=600,
           layer_range=["max_pooling2d","activation_3"],
           rankdir='TB',
           show_layer_activations=True,
           expand_nested=False)
