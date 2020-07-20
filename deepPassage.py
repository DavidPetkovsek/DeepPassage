# import StyleGan2Interface as sg
# import dataloader

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from stylegan.stylegan_two import StyleGAN as StyleGAN2

# output size should be 512 for StyleGan2

# https://towardsdatascience.com/transfer-learning-with-tf-2-0-ff960901046d
IMG_HEIGHT = 256
IMG_WIDTH = IMG_HEIGHT
IMG_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 1
EVALUATION_INTERVAL = 200

class DeepPassage(object):
    def __init__(self):
        # https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG19
        # base_model = tf.keras.applications.ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_top=False, weights='imagenet')
        base_model = tf.keras.applications.vgg19.VGG19(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_top=False, weights='imagenet')
        base_model.trainable = False

        x = base_model.output
        x = layers.Conv2D(512, 3, padding='same',activation='relu')(x)
        x = layers.Conv2D(512, 3, padding='same',activation='relu')(x)
        x = layers.Reshape((512, 8*8))(x)
        x = layers.Conv1D(8,1,padding='same',activation='relu')(x)
        xs = tf.split(x, 8, axis=2)
        xs[-1] = layers.LSTM(512)(xs[-1])
        xs[-1] = layers.Reshape((512, 1))(xs[-1]) # the input of the next layers need the extra dimension
        xs[-1] = layers.LSTM(256)(xs[-1])
        xs[-1] = layers.Reshape((16, 16, 1))(xs[-1]) # the input of the next layers need the extra dimension
        xs[-1] = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(xs[-1])
        xs[-1] = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(xs[-1])
        xs[-1] = layers.Conv2DTranspose(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(xs[-1])
        xs[-1] = layers.Reshape((256, 256, 1))(xs[-1]) # reshape to proper shape for StyleGAN2
        xs[0] = layers.Reshape([512])(xs[0]) # reshape to proper shape for StyleGAN2
        xs[1] = layers.Reshape([512])(xs[1]) # reshape to proper shape for StyleGAN2
        xs[2] = layers.Reshape([512])(xs[2]) # reshape to proper shape for StyleGAN2
        xs[3] = layers.Reshape([512])(xs[3]) # reshape to proper shape for StyleGAN2
        xs[4] = layers.Reshape([512])(xs[4]) # reshape to proper shape for StyleGAN2
        xs[5] = layers.Reshape([512])(xs[5]) # reshape to proper shape for StyleGAN2
        xs[6] = layers.Reshape([512])(xs[6]) # reshape to proper shape for StyleGAN2

        sg = StyleGAN2()
        # sg.load(28)
        x = sg.GAN.G(xs)
        sg.GAN.G.trainable = False
        for i in range(len(sg.GAN.G.layers)):
            sg.GAN.G.layers[i].trainable = False
        self.full_model = models.Model(inputs=base_model.input, outputs=x)

        self.evalModel = tf.keras.applications.vgg19.VGG19(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_top=False, weights='imagenet')
        self.evalModel.trainable = False
        self.losses = []

    def train(self):
        loss = self.train_step()


    @tf.function
    def train_step(self, image, next_image):
        with tf.GradientTape() as tape:
            generated_image = self.full_model(image)
            loss = keras.losses.MSE(evalModel(next_image), evalModel(generated_image))
        gradient = tape.gradient(loss, self.full_model.trainable_variables)
        self.full_model.apply_gradients(zip(gradient, self.full_model.trainable_variables))
        return loss

if __name__ == '__main__':
    dp = DeepPassage()
        # print(full_model.summary())
        # exit()
        #
        #
        # evalModel = tf.keras.applications.vgg19.VGG19(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_top=False, weights='imagenet')
        # evalModel.trainable = False
        # def vgg_loss(expected, prediction):
        #     return keras.losses.MSE(evalModel(expected), evalModel(prediction))
        # full_model.compile(optimizer=keras.optimizers.RMSprop(clipvalue=1.0), loss=vgg_loss)
        # model.fit(train_data_multi, epochs=EPOCHS,
        #                           steps_per_epoch=EVALUATION_INTERVAL,
        #                           validation_data=val_data_multi,
        #                           validation_steps=50)
