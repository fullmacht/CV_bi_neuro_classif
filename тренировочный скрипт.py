import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg16

import os
from random import shuffle
from glob import glob

IMG_SIZE = (224, 224)  # размер входного изображения сети

train_files = glob('C:/Users/pc/Desktop/train/*.jpg')
test_files = glob('C:/Users/pc/Desktop/test/*.jpg')

# загружаем входное изображение и предобрабатываем
def load_image(path, target_size=IMG_SIZE):
    img = cv2.imread(path)[...,::-1]
    img = cv2.resize(img, target_size)
    return vgg16.preprocess_input(img)  # предобработка для VGG16

# функция-генератор загрузки обучающих данных с диска
def fit_generator(files, batch_size=32):
    batch_size = min(batch_size, len(files))
    while True:
        shuffle(files)
        for k in range(len(files) // batch_size):
            i = k * batch_size
            j = i + batch_size
            if j > len(files):
                j = - j % len(files)
            x = np.array([load_image(path) for path in files[i:j]])
            y = np.array([1. if os.path.basename(path).startswith('male') else 0.
                          for path in files[i:j]])
            yield (x, y)


def predict_generator(files):
    while True:
        for path in files:
            yield np.array([load_image(path)])

base_model = vgg16.VGG16(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.layers[-5].output
x = tf.compat.v1.keras.layers.BatchNormalization(
    axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones',
    beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
    fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None,
)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1,  # один выход (бинарная классификация)
                          activation='sigmoid',  # функция активации
                          kernel_regularizer=tf.keras.regularizers.l1(1e-4))(x)

model = tf.keras.Model(inputs=base_model.input, outputs=x, name='male-female')

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # функция потерь binary_crossentropy (log loss
              metrics=['accuracy'])

val_samples = 5  # число изображений в валидационной выборке

shuffle(train_files)  # перемешиваем обучающую выборку
validation_data = next(fit_generator(train_files[:val_samples], val_samples))
train_data = fit_generator(train_files[val_samples:])  # данные читаем функцией-генератором

# запускаем процесс обучения
model.fit(train_data,
          steps_per_epoch=10,  # число вызовов генератора за эпоху
          epochs=100,  # число эпох обучения
          validation_data=validation_data)

model.save('male-female.hdf5')

test_pred = model.predict(
    predict_generator(test_files), steps=len(test_files))

import re
# сохраняем в файл
with open('submit.txt', 'w') as dst:
    dst.write('id,label\n')
    for path, score in zip(test_files, test_pred):
        dst.write('%s,%f\n' % (re.search('(\d+).jpg$', path).group(1), score))

import pandas as pd
sub_read = pd.read_csv(r'C:\Users\pc\PycharmProjects\N_tech_lab\submit.txt')
sub_read.to_csv(r'C:\Users\pc\PycharmProjects\N_tech_lab\submit.csv',index=None,sep=',')

data = pd.read_csv(r'C:\Users\pc\PycharmProjects\N_tech_lab\submit.csv')
# Меняем значения валидационной метрики на значения male, female
test_val = []
i = 0
for val in data['label']:
    if val > 0.5:
        val = 'male'
        test_val.append(val)
    else:
        val = 'female'
        test_val.append(val)

data['label'] = test_val

id_list = []
for id in data['id']:
    id = 'img_{}.jpg'.format(id)
    id_list.append(id)
data['id']= id_list

data = data.set_index(data['id'])
data = data.drop('id',axis=1)
# Сохраняем json
data.to_json(r'C:\Users\pc\PycharmProjects\N_tech_lab\process_result.json',)