import os
#
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras

from Utils import preprocessing, metrics
import Model

EVALUATE_PARAM = {
    'data_path': './DataSet/Validation',
    'model_path': './Project/model-lt-resnet.h5',
    'batch_size': 8
}


def evaluate_project(source_model: str):

    model = keras.models.load_model(source_model)

    data_p = preprocessing.Preprocessing(path=EVALUATE_PARAM['data_path'],
                                         size=model.input_shape[1:-1],
                                         batch_size=EVALUATE_PARAM['batch_size'])
    data_generator, steps = data_p.get_data_generator(pre=False, shuffle=False)

    model.compile(loss=Model.losses.focal_loss(),
                  metrics=[metrics.recall,
                           metrics.precision,
                           metrics.f1_score,
                           metrics.f2_score])

    _metrics = ['loss', 'recall', 'precision', 'f1_score', 'f2_score']
    _evaluate = model.evaluate_generator(generator=data_generator, steps=steps, verbose=0)

    print('*' * 10, 'evaluate')
    for m, e in zip(_metrics, _evaluate):
        print(f'{m} = {e}')
    print('*' * 10, 'end')
    #################### end

    data_p.close_generator()
    del model


#################### 评估 ####################
if __name__ == '__main__':
    evaluate_project(EVALUATE_PARAM['model_path'])

