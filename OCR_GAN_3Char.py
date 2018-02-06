


from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU

from sys import getdefaultencoding
import sys

d = getdefaultencoding()
if d != "utf-8":
    reload(sys)
    sys.setdefaultencoding("utf-8")
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
import gzip,cPickle

import matplotlib.pyplot as plt

def dataset_load(path):
    if path.endswith(".gz"):
        f=gzip.open(path,'rb')
    else:
        f=open(path,'rb')

    if sys.version_info<(3,):
        data=cPickle.load(f)
    else:
        data=cPickle.load(f,encoding="bytes")
    f.close()
    return data

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, img_rows=564, img_cols=64, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def discriminator(self):
        # img_w, img_h = 564, 64
        img_w, img_h = 128, 32
        words_per_epoch = 16000
        val_split = 0.2
        val_words = int(words_per_epoch * (val_split))
        # 1, 32, 128

        # Network parameters
        conv_filters = 16
        kernel_size = (3, 3)
        pool_size = 2
        time_dense_size = 32
        rnn_size = 512
        minibatch_size = 32
        act = 'relu'

        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_w, img_h)
        else:
            input_shape = (img_h, img_w, 1)

        input_data = Input(name='the_input', shape=input_shape, dtype='float32')
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

        conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

        # Two layers of bidirectional GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
            inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
            gru1_merged)

        inner = Dense(327, kernel_initializer='he_normal',
                      name='dense2')(concatenate([gru_2, gru_2b]))
        inner_act = Activation('sigmoid', name='Sigmoid_1')(inner)

        inner_act = Flatten()(inner_act)

        inner_2 = Dense(4,
                        name='dense3')(inner_act)
        inner_out = Activation('softmax')(inner_2)

        model = Model(inputs=input_data, outputs=inner_out)

        model.summary()

        return model

    def generator(self):
        G = Sequential()
        dropout = 0.4
        depth = 64 * 4
        dim1 = 32
        dim2 = 8
        # Input = 164 + 1 = 165
        G.add(Dense(dim1 * dim2 * depth, input_dim=164))
        G.add(BatchNormalization(momentum=0.9))
        G.add(Activation('relu'))
        G.add(Reshape((dim2, dim1, depth)))
        G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        G.add(UpSampling2D())
        G.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
        G.add(BatchNormalization(momentum=0.9))
        G.add(Activation('relu'))

        G.add(UpSampling2D())
        G.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
        G.add(BatchNormalization(momentum=0.9))
        G.add(Activation('relu'))

        G.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
        G.add(BatchNormalization(momentum=0.9))
        G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        G.add(Conv2DTranspose(1, 5, padding='same')) # Outputs a 564x64 layer output
        G.add(Activation('sigmoid'))
        G.summary()
        return G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='categorical_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        # self.AM.add(self.discriminator())
        self.AM.add(Dense(1, activation='softmax'))
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

class OCR_DCGAN(object):
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 32
        self.channel = 1
        import wordgenerators_sequential as wm
        self.x_train,self.y_train = wm.CreateDataset()
        self.x_train = self.x_train.reshape(-1, 32,128 , 1).astype(np.float32)
        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 164])
        for i in range(train_steps):
            X=[]
            Y=[]
            for i in range(batch_size):

                rand = np.random.randint(0,self.x_train.shape[0])
                X.append(self.x_train[rand])
                Y.append(self.y_train[rand])

            print(X[0].shape)
            # #Training The discriminator First
            # images_train = self.x_train[np.random.randint(0,
            #     self.x_train.shape[0], size=batch_size), :, :, :], # Select Random Images from OCR Images
            X=np.asarray(X)

            print(X.shape)
            Y=np.asarray(Y)

            # Making of "Shob" in Bangla
            fakelabel=[]
            basicnoise=[1,2]
            for i in range(3,165):
                basicnoise.append(4)
            noise=[]
            for i in range(batch_size):
                noise.append(basicnoise)
                fakelabel.append([1,2,0,0])

            noise = np.array(noise)
            print(noise.shape)
            images_fake = self.generator.predict(noise) # Get the fake images from the generatro using noises
            print(X.shape," - ", images_fake.shape)
            x = np.concatenate((X, images_fake)) # Merge the original and fake images
            y = np.concatenate((Y,fakelabel))
            # y = np.ones([2*batch_size, 1]) # Label The merged images together
            # y[batch_size:, :] = 0 # Make the fake images label as 0
            d_loss = self.discriminator.train_on_batch(x, y) # Train the discriminator

            #Adversarial Training
            y = np.ones([batch_size, 1])

            basicnoise = [1, 2]
            for i in range(3, 165):
                basicnoise.append(4)
            noise = []
            for i in range(batch_size):
                noise.append(basicnoise)
            noise = np.array(noise)

            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'ocr.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 165])
            else:
                filename = "ocr_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.generator()
    ocr_dcgan = OCR_DCGAN()
    timer = ElapsedTimer()
    ocr_dcgan.train(train_steps=10, batch_size=256, save_interval=500)
    timer.elapsed_time()
    ocr_dcgan.plot_images(fake=True)
ocr_dcgan.plot_images(fake=False, save2file=True)