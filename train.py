import datetime

import numpy as np
import matplotlib.pyplot as plt
import os

#
import Model.losses

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
#
from Model import resnet
from Utils import preprocessing, metrics, label_code

TRAIN_PARAM = {
    # 模型
    'target_size': (128, 128),
    'model_type': 'lt_resnet_152',
    # 训练
    'optimizer_type': 'adam',
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 2,
    'epochs_period': 10,
    'use_mixed_precision': True,
    # 路径
    'train_path': './DataSet/Train',
    'validation_path': './DataSet/Validation',
    'model_path': './Project',
    'checkpoint_path': './Project/checkpoints',
}


def train_project(source_model=None):
    optimizer_type = TRAIN_PARAM['optimizer_type']
    learning_rate = TRAIN_PARAM['learning_rate']
    use_mixed_precision = TRAIN_PARAM['use_mixed_precision']
    model_type = TRAIN_PARAM['model_type']

    if optimizer_type == 'adam':
        optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0)
        print('Using Adam optimizer ... ')
    elif optimizer_type == 'sgd':
        optimizer = tf.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)
        print('Using SGD optimizer ... ')
    elif optimizer_type == 'rmsprop':
        optimizer = tf.optimizers.RMSprop(learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
        print('Using RMSprop optimizer ... ')
    else:
        raise Exception(
            f"Not supported optimizer : {optimizer_type}")

    if use_mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        print('Optimizer is changed to LossScaleOptimizer.')

    if model_type == 'resnet_152':
        model_delegate = resnet.create_resnet_152
    elif model_type == 'lt_resnet_152':
        model_delegate = resnet.create_lt_resnet_152
    elif model_type == 'resnet_custom_v1':
        model_delegate = resnet.create_resnet_custom_v1
    elif model_type == 'resnet_custom_v2':
        model_delegate = resnet.create_resnet_custom_v2
    elif model_type == 'resnet_custom_v3':
        model_delegate = resnet.create_resnet_custom_v3
    elif model_type == 'resnet_custom_v4':
        model_delegate = resnet.create_resnet_custom_v4
    else:
        raise Exception(f'Not supported model : {model_type}')

    if source_model:
        pass
    else:
        os.mkdir('Project')
        os.mkdir('Project/checkpoints')
        os.mkdir('Project/log')
        print('MKDIR SUSS')

    #################### 图像预处理生成器
    train_p = preprocessing.Preprocessing(path=TRAIN_PARAM['train_path'],
                                          size=TRAIN_PARAM['target_size'],
                                          batch_size=TRAIN_PARAM['batch_size'])
    validation_p = preprocessing.Preprocessing(path=TRAIN_PARAM['validation_path'],
                                               size=TRAIN_PARAM['target_size'],
                                               batch_size=TRAIN_PARAM['batch_size'])

    train_generator, steps_per_epoch = train_p.get_data_generator(pre=True)
    validation_generator, validation_steps = validation_p.get_data_generator(pre=False)
    #################### end

    #################### 构建模型

    if source_model:
        model = keras.models.load_model(source_model, )
        # custom_objects={
        #     'recall': metrics.recall,
        #     'precision': metrics.precision,
        #     'f1_score': metrics.f1_score,
        #     'f2_score': metrics.f2_score,
        # })
        print(f'Model: {model.input_shape} -> {model.output_shape} (loaded from {source_model})')
    else:
        inputs = keras.Input(shape=(*TRAIN_PARAM['target_size'], 3), dtype=np.float32)
        ouputs = model_delegate(inputs, output_dim=len(train_p.get_label_name()))
        model = keras.Model(inputs=inputs, outputs=ouputs, name=model_type)

    model.compile(optimizer=optimizer,
                  #loss=keras.losses.CategoricalCrossentropy(),
                  loss=Model.losses.focal_loss(),
                  metrics=[metrics.recall,
                           metrics.precision,
                           metrics.f1_score,
                           metrics.f2_score])
    #################### end

    #################### 训练模型
    point_name = "model_{epoch:02d}-{loss:.2f}-{recall:.2f}-{precision:.2f}-{f1_score:.2f}-{f2_score:.2f}.h5"

    #Tensorboard
    log_dir = 'Project/log'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


    checkpoint = ModelCheckpoint(os.path.join(TRAIN_PARAM['checkpoint_path'], point_name),
                                 monitor='loss',
                                 save_weights_only=True,
                                 #period=TRAIN_PARAM['epochs_period'],#`period` argument is deprecated. Please use `save_freq` to specify the frequency in number of batches seen.
                                 save_freq=TRAIN_PARAM['epochs_period'],
                                 verbose=1, )
    with tf.device('/device:GPU:0'):
        history = model.fit(train_generator,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=validation_generator,
                            validation_steps=validation_steps,
                            epochs=TRAIN_PARAM['epochs'],
                            verbose=1,
                            callbacks=[checkpoint, tensorboard_callback])
    #################### endcallbacks=[checkpoint],

    #################### 保存模型以及数据
    model_name = f"model-{model_type}.h5"
    label_name = f"model-{model_type}.txt"
    plt_name = f"model-{model_type}.jpg"

    model.compile()
    model.save(os.path.join(TRAIN_PARAM['model_path'], model_name))

    label = train_p.get_label_name()
    label_code.save_label(path=os.path.join(TRAIN_PARAM['model_path'], label_name),
                          labels=label)

    save_plt(os.path.join(TRAIN_PARAM['model_path'], plt_name), history)
    #################### end

    train_p.close_generator()
    validation_p.close_generator()
    del model


def save_plt(path: str, history):
    history: dict = history.history
    epochs = range(1, len(history['loss']) + 1)

    color1 = '#F59340'
    color2 = '#40D5F5'

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.title("Training and Validation [loss]")
    plt.plot(epochs, history['loss'], '-', label="Training loss", alpha=0.9, color=color1)
    plt.plot(epochs, history['val_loss'], 'o', label="Validation loss", alpha=0.9, color=color1)
    plt.ylabel('loss')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.title("Training and Validation [recall/precision]")
    plt.plot(epochs, history['recall'], '-', label="Training recall", alpha=0.9, color=color1)
    plt.plot(epochs, history['val_recall'], 'o', label="Validation recall", alpha=0.9, color=color1)
    plt.plot(epochs, history['precision'], '-', label="Training precision", alpha=0.9, color=color2)
    plt.plot(epochs, history['val_precision'], 'o', label="Validation precision", alpha=0.9, color=color2)
    plt.ylabel('R / P')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.title("Training and Validation [f1_score/f2_score]")
    plt.plot(epochs, history['f1_score'], '-', label="Training f1_score", alpha=0.9, color=color1)
    plt.plot(epochs, history['val_f1_score'], 'o', label="Validation f1_score", alpha=0.9, color=color1)
    plt.plot(epochs, history['f2_score'], '-', label="Training f2_score", alpha=0.9, color=color2)
    plt.plot(epochs, history['val_f2_score'], 'o', label="Validation f2_score", alpha=0.9, color=color2)
    plt.ylabel('f_score')
    plt.xlabel('epochs')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(path)


#################### 训练 ####################
if __name__ == '__main__':
    # 在已有的模型上进行训练
    #train_project('./Project/model-resnet_152.h5')

    # 重新训练一个新的模型
    train_project()
