import os
import fnmatch
import cv2
import numpy as np
import string
import time

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tqdm import tqdm

# ignore warnings in the output
# tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.python.client import device_lib

# Check all available devices if GPU is available
print(device_lib.list_local_devices())
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# char_list:   'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# total number of our output classes: len(char_list)
char_list = string.ascii_letters + string.digits


def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
    return dig_lst


path = 'C:/Data\mjsynth/mnt/ramdisk/max/90kDICT32px'

# lists for training dataset
training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

# lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

max_label_len = 0

i = 1
flag = 0

for root, dirnames, filenames in tqdm(os.walk(path),desc='folders'):

    for f_name in fnmatch.filter(filenames, '*.jpg'):
        # read input image and convert into gray scale image
        img = cv2.cvtColor(cv2.imread(os.path.join(root, f_name)), cv2.COLOR_BGR2GRAY)

        # convert each image of shape (32, 128, 1)
        w, h = img.shape
        if h > 128 or w > 32:
            continue
        if w < 32:
            add_zeros = np.ones((32 - w, h)) * 255
            img = np.concatenate((img, add_zeros))

        if h < 128:
            add_zeros = np.ones((32, 128 - h)) * 255
            img = np.concatenate((img, add_zeros), axis=1)
        img = np.expand_dims(img, axis=2)

        # Normalize each image
        img = img / 255.

        # get the text from the image
        txt = f_name.split('_')[1]

        # compute maximum length of the text
        if len(txt) > max_label_len:
            max_label_len = len(txt)

        # split the 150000 data into validation and training dataset as 10% and 90% respectively
        if i % 10 == 0:
            valid_orig_txt.append(txt)
            valid_label_length.append(len(txt))
            valid_input_length.append(31)
            valid_img.append(img)
            valid_txt.append(encode_to_labels(txt))
        else:
            orig_txt.append(txt)
            train_label_length.append(len(txt))
            train_input_length.append(31)
            training_img.append(img)
            training_txt.append(encode_to_labels(txt))

            # break the loop if total data is 150000
        if i == 150000:
            flag = 1
            break
        i += 1
    if flag == 1:
        break


# pad each output label to maximum text length

train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value=len(char_list))
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value=len(char_list))




# prepare model architecture

# input with shape of height=32 and width=128
inputs = Input(shape=(32, 128, 1))

# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
# poolig layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)

conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)
act_model.summary()


labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

# model to be used at training time
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)



model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

filepath = "best_model_.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
early = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', cooldown=0,min_lr=0.000000001)
csv = CSVLogger('trainingLossLogFile.csv')
callbacks_list = [checkpoint, early, lr, csv]

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)


batch_size = 128
epochs = 10
model.load_weights('best_model_.hdf5', by_name=True)
model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length], y=np.zeros(len(training_img)),
          batch_size=batch_size, epochs=epochs, validation_data=(
    [valid_img, valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(len(valid_img))]), verbose=1,
          callbacks=callbacks_list)




# load the saved best model weights
act_model.load_weights('crnn_model.hdf5') # original trained model
act_model.load_weights('better_trained_model.h5') # better_trained_model on local gpu with cropped data from synthtext dataset


# predict outputs on validation images
#create a test dataset
def prepare_image(path):
    test_im = []
    for root, dirnames, filenames in tqdm(os.walk(path), desc='folders'):
        print('222')
        for f_name in filenames:
            if f_name.endswith(('.png', '.jpg', '.jpeg')):
                # read input image and convert into gray scale image
                im = cv2.cvtColor(cv2.imread(os.path.join(root, f_name)), cv2.COLOR_BGR2GRAY)
                w,h = im.shape

                if h > 128 or w > 32:
                   continue
                    # im = cv2.resize(im, (32,128))
                    # w, h = im.shape

                if w < 32:
                    add_zeros = np.ones((32 - w,h)) * 255
                    im = np.concatenate((im,add_zeros))

                if h < 128:
                    add_zeros = np.ones((32,128-h))*255
                    im = np.concatenate((im,add_zeros), axis=1)
                im = np.expand_dims(im,axis=2)

                # normalize the image
                im = im/255.
                test_im.append(im)

    return np.array(test_im)


test_path = './test_data'
test_im = prepare_image(test_path)
gt_file = open('./test-images-gt.txt', "r").readlines()
label =  [name.split('__')[1] for name in gt_file]
prediction = act_model.predict(test_im)

# use CTC decoder
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                               greedy=True)[0][0])


# see the results
i = 0
for x in out:
    print("predicted text = ", end='')
    for p in x:
        if int(p) != -1:
            print(char_list[int(p)], end='')
    print('\n')
    cv2.imshow('img',test_im[i])
    cv2.waitKey(0)
    cv2.imwrite()
    i += 1

