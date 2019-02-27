"""
Auxiliary classifier DCGAN for pecan dataset using Keras
"""

import datetime

import keras
import numpy as np
from keras.layers import Input, Conv1D, LeakyReLU, BatchNormalization, Flatten, Dense, Lambda
from keras.layers import Reshape, Conv2DTranspose, Activation, Embedding, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data import date_format_day
from metric import mmd_loss


class ACGAN(object):
    def __init__(self, input_dim, window_length, weight_path, code_size=64, learning_date=1e-4,
                 batch_size=32):
        self.input_dim = input_dim
        self.code_size = code_size
        assert window_length % 8 == 0, 'This DCGAN architecture requires window length to be multiple of 8'
        self.window_length = window_length
        self.learning_rate = learning_date
        self.batch_size = batch_size
        self.weight_path = weight_path

        self.generator = self._create_generator()
        self.discriminator = self._create_discriminator()
        self.discriminator_generator = self._combine_generator_discriminator()

    def _create_generator(self):
        final_window_length = int(self.window_length / 8)
        noise = Input(shape=(self.code_size,))
        month_label_input = Input(shape=(1,), dtype='int32')
        day_label_input = Input(shape=(1,), dtype='int32')
        self.month_embedding_layer = Embedding(input_dim=12, output_dim=self.code_size)
        self.day_embedding_layer = Embedding(input_dim=7, output_dim=self.code_size)
        month_embedding = Flatten()(self.month_embedding_layer(month_label_input))
        day_embedding = Flatten()(self.day_embedding_layer(day_label_input))
        x = Concatenate(axis=-1)([noise, month_embedding, day_embedding])
        x = Dense(final_window_length * 64)(x)
        x = Reshape(target_shape=(final_window_length, 1, 64))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(filters=32, kernel_size=(4, 1), strides=(2, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(filters=16, kernel_size=(4, 1), strides=(2, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(filters=self.input_dim, kernel_size=(4, 1), strides=(2, 1), padding='same')(x)
        x = Lambda(lambda x: keras.backend.squeeze(x, axis=-2))(x)
        output = Activation('sigmoid')(x)
        model = Model(inputs=[noise, month_label_input, day_label_input], outputs=output)
        model.compile(optimizer=Adam(lr=self.learning_rate, beta_1=0.5), loss='binary_crossentropy')
        return model

    def _create_discriminator(self):
        time_series_input = Input(shape=(self.window_length, self.input_dim))
        x = Conv1D(filters=16, kernel_size=4, strides=2, padding='same')(time_series_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv1D(filters=32, kernel_size=4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv1D(filters=64, kernel_size=4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        fake = Dense(1, activation='sigmoid')(x)
        month_label_output = Dense(12, activation='softmax')(x)
        day_label_output = Dense(7, activation='softmax')(x)
        model = Model(inputs=time_series_input, outputs=[fake, month_label_output, day_label_output])
        model.compile(optimizer=Adam(lr=self.learning_rate, beta_1=0.5, decay=1e-6),
                      loss=['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'])
        return model

    def _combine_generator_discriminator(self):
        latent = Input(shape=(self.code_size,))
        month_label_input = Input(shape=(1,), dtype='int32')
        day_label_input = Input(shape=(1,), dtype='int32')
        generated_data = self.generator([latent, month_label_input, day_label_input])
        self.discriminator.trainable = False
        fake, month_label_output, day_label_output = self.discriminator(generated_data)
        combined = Model([latent, month_label_input, day_label_input], [fake, month_label_output, day_label_output])
        combined.compile(optimizer=Adam(lr=self.learning_rate, beta_1=0.5, decay=1e-6),
                         loss=['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'])
        return combined

    def train(self, x_train, x_val, num_epoch=5):
        summary_writer = SummaryWriter()
        self.gen_losses = []
        self.dis_losses = []
        self.mmd_losses = []

        train_samples, month_label, day_label = x_train
        num_train = train_samples.shape[0]
        step = 0

        index_array = np.arange(num_train)
        validation_data = x_val

        for epoch in range(num_epoch):
            np.random.shuffle(index_array)
            for i in tqdm(range(num_train // self.batch_size), desc='Epoch {}: '.format(epoch + 1)):
                current_index = index_array[i * self.batch_size: (i + 1) * self.batch_size]
                # get image
                time_series_batch = train_samples[current_index]
                # get label
                month_label_batch = month_label[current_index]
                day_label_batch = day_label[current_index]
                # get noise
                noise = np.random.normal(0., 1., [self.batch_size, self.code_size])
                # get a batch of fake images
                generated_time_series = self.generator.predict(
                    [noise, month_label_batch.reshape((-1, 1)), day_label_batch.reshape((-1, 1))], verbose=0)
                soft_zero, soft_one = 0, 0.95

                dis_loss_image = self.discriminator.train_on_batch(time_series_batch,
                                                                   [np.array([soft_one] * self.batch_size),
                                                                    to_categorical(month_label_batch, 12),
                                                                    to_categorical(day_label_batch, 7)])
                dis_loss_noise = self.discriminator.train_on_batch(
                    generated_time_series, [[soft_zero] * self.batch_size, to_categorical(month_label_batch, 12),
                                            to_categorical(day_label_batch, 7)])
                dis_loss = dis_loss_image + dis_loss_noise

                dis_loss = np.sum(dis_loss)
                # print(dis_loss)
                noise = np.random.normal(0., 1., (2 * self.batch_size, self.code_size))
                # get sample labels
                month_sampled_labels = np.random.randint(0, 12, 2 * self.batch_size)
                day_sampled_labels = np.random.randint(0, 7, 2 * self.batch_size)
                trick = np.ones(2 * self.batch_size) * soft_one
                gen_loss = self.discriminator_generator.train_on_batch(
                    [noise, month_sampled_labels, day_sampled_labels],
                    [trick, to_categorical(month_sampled_labels, 12), to_categorical(day_sampled_labels, 7)])

                gen_loss = np.sum(gen_loss)

                summary_writer.add_scalars('data/train_loss', {'gen': gen_loss,
                                                               'dis': dis_loss},
                                           global_step=step)

                step += 1

            # sample a batch and calculate mmd loss with validation data
            x_val = validation_data[0]
            y_val = validation_data[1:3]
            x_generated = self.generate(y_val)
            mmd_loss_vec = np.zeros(shape=(x_val.shape[-1]))
            for j in range(x_val.shape[-1]):
                mmd_loss_vec[j] = mmd_loss(x_val[:, :, j], x_generated[:, :, j], weight=1.)
            summary_writer.add_scalars('data/mmd_loss', {'load': mmd_loss_vec[0],
                                                         'pv': mmd_loss_vec[1]},
                                       global_step=epoch)
        self.save_weight()

    def _generate(self, x):
        return self.generator.predict(x)

    def generate(self, labels):
        num_samples = labels[0].shape[0]
        z = np.random.normal(0, 1, size=[num_samples, self.code_size])
        return self._generate([z] + labels)

    def generate_by_date(self, num_samples, starting_date_str='2013-01-01'):
        month_labels = np.zeros(shape=(num_samples))
        day_labels = np.zeros(shape=(num_samples))
        starting_date = datetime.datetime.strptime(starting_date_str, date_format_day)
        for i in range(num_samples):
            current_date = starting_date + datetime.timedelta(i)
            month_labels[i] = current_date.month - 1
            day_labels[i] = current_date.weekday()
        return self.generate([month_labels, day_labels])

    def save_weight(self):
        self.generator.save_weights(self.weight_path + '_acgan_generator.h5')
        self.discriminator.save_weights(self.weight_path + '_acgan_discriminator.h5')

    def load_weight(self):
        self.generator.load_weights(self.weight_path + '_acgan_generator.h5')
        self.discriminator.load_weights(self.weight_path + '_acgan_discriminator.h5')
