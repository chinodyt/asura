import keras as Keras
from keras import layers
from keras import models
from keras import backend as K
from keras import callbacks
import tensorflow as tf


import sys
import numpy as np
import cv2

import time

import util




def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

class UNet() :
    def __init__(self, input_shape = (512,512,3), num_class = 3):
        self.model = None
        self.input_shape = input_shape
        self.create_model(num_class=num_class)

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)


    def create_model(self, num_class=3) :
        concat_axis = 3
        inputs = layers.Input(shape = self.input_shape)
        
        s = layers.core.Lambda(lambda x: x / 255) (inputs)

        c1 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
        c1 = layers.Dropout(0.1) (c1)
        c1 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
        p1 = layers.MaxPooling2D((2, 2)) (c1)

        c2 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
        c2 = layers.Dropout(0.1) (c2)
        c2 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
        p2 = layers.MaxPooling2D((2, 2)) (c2)

        c3 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
        c3 = layers.Dropout(0.2) (c3)
        c3 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
        p3 = layers.MaxPooling2D((2, 2)) (c3)

        c4 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
        c4 = layers.Dropout(0.2) (c4)
        c4 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
        c5 = layers.Dropout(0.3) (c5)
        c5 = layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

        u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
        c6 = layers.Dropout(0.2) (c6)
        c6 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

        u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
        c7 = layers.Dropout(0.2) (c7)
        c7 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

        u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
        c8 = layers.Dropout(0.1) (c8)
        c8 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

        u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
        c9 = layers.Dropout(0.1) (c9)
        c9 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

        outputs = layers.Conv2D(num_class, (1, 1), activation='sigmoid') (c9)

        self.model = models.Model(inputs = inputs, output = outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

    def create_model_filename(self, dataset, model) :
        saveTo = './models/%s_%s_E%d_weights.h5' % (dataset.dataset_name, model['model_type'], model['epochs'])
        return saveTo

    def create_output_path(self, dataset, model) :
        ds_name = dataset.dataset_name[:-3]
        output_path = './output/%s/%s_E%d' % (ds_name, model['model_type'], model['epochs'])
        return output_path

    def load(self, loadFrom = '../models/unet_weights.h5') :
        try:
            self.model.load_weights(loadFrom)
            print 'Loaded.'
            return True
        except:
            print 'Failed to load.'
            print sys.exc_info()[0]
            return False


    def train(self, trainPath = None, trainset = None, saveTo = '../models/unet_weights.h5', model_args = None):
        print 'Training!!'
        if trainset is None :
            trainset = util.ImageDataset(trainPath)

        batch_size = 16
        epochs = 500
        if model_args is not None :
            if 'batch_size' in model_args :
                batch_size = model_args['batch_size']
            if 'epochs' in model_args :
                epochs = model_args['epochs']

        imgList = trainset.imgList
        maskList = trainset.maskList
        
        print 'Treating output masks...'
        maskList = self.treat_mask(maskList)
        print 'Done treating.'

        print imgList.shape
        print maskList.shape
        earlystopper = callbacks.EarlyStopping(patience=5, verbose=1)
        checkpointer = callbacks.ModelCheckpoint(saveTo, verbose=1, save_best_only=True)
        results = self.model.fit(imgList, maskList, batch_size=batch_size, epochs=epochs, 
                                callbacks=[earlystopper, checkpointer])
        
        self.model.save_weights(saveTo)
        print self.model.summary()

    def treat_mask(self, mask_list) :
        new_list = []
        
        
        total = len(mask_list)
        for i in xrange(len(mask_list)) :
            print '%d/%d... \r' % (i, total)
            new_mask = util.treat_mask(mask_list[i])

            new_list.append(new_mask)

        return np.array(new_list)

    def treat_output(self, ret) :
        m_width = ret.shape[0]
        m_height = ret.shape[1]
        new_mask = np.zeros((m_width, m_height, 3))
        new_mask[ret[:,:,0] > 0.5] = [255,255,255]
        new_mask[ret[:,:,1] > 0.5] = [0,0,255]



        return new_mask

    def run(self, image, filename = None) :
        imgRsz = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        ret = self.model.predict(imgRsz.reshape(1,self.input_shape[0],self.input_shape[1],3))
        return self.treat_output(cv2.resize(ret[0], (image.shape[1],image.shape[0])))

def __test1__() :
    args = util.Settings()
    trainPath = args.settings['datasets']['train']

    print 'Loading images'
    trainset = util.ImageDataset(trainPath)
    imgList, maskList = trainset.get_images()

    print 'Creating U-Net'
    model = UNet()

    if not model.load() :
        print 'Creating and Training'
        model.train(trainset = trainset)
    else :
        print 'Loaded'
    print model.model.summary()
    print imgList[0].shape
    ret = model.run(trainset.imgList[0])
    print ret.shape
    cv2.imshow('gota',ret)
    cv2.waitKey(0)



if __name__ == '__main__':
    __test1__()


