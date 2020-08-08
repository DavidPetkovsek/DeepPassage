# import StyleGan2Interface as sg
# import dataloader

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from stylegan.stylegan_two import StyleGAN as StyleGAN2
import os
import cv2
import random
from tqdm import tqdm
import numpy as np
import shutil

tfds = tf.data.Dataset

# output size should be 512 for StyleGan2

# https://towardsdatascience.com/transfer-learning-with-tf-2-0-ff960901046d
IMG_HEIGHT = 256
IMG_WIDTH = IMG_HEIGHT
IMG_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 1000
FRAME_PER_CLIP = 25
TEST_SAMPLE = 200
EPOCH_LENGTH = 200

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
        sg.load(0)
        x = sg.GAN.G(xs)
        sg.GAN.G.trainable = False
        for i in range(len(sg.GAN.G.layers)):
            sg.GAN.G.layers[i].trainable = False
        self.full_model = models.Model(inputs=base_model.input, outputs=x)

        self.evalModel = tf.keras.applications.vgg19.VGG19(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_top=False, weights='imagenet')
        self.evalModel.trainable = False
        self.losses = []
        self.optimizer = keras.optimizers.RMSprop(clipvalue=1.0)

    def train(self, trainingset, testset):
        for samples in tqdm(trainingset, total=1500):
            frames, next_frames = tf.split(samples, 2, axis=1)
            frames = tf.squeeze(frames, axis=1)
            next_frames = tf.squeeze(next_frames, axis=1)
            loss = self.train_step(frames, next_frames)
            self.losses.append(loss)
        loss = 0
        for samples in testset:
            frames, next_frames = tf.split(samples, 2, axis=1)
            frames = tf.squeeze(frames, axis=1)
            next_frames = tf.squeeze(next_frames, axis=1)
            loss += keras.losses.MSE(self.evalModel(next_frames), self.evalModel(self.full_model(frames)))
        return loss

    @tf.function
    def train_step(self, image, next_image):
        with tf.GradientTape() as tape:
            generated_image = self.full_model(image)
            loss = keras.losses.MSE(self.evalModel(next_image), self.evalModel(generated_image))
            gradients = tape.gradient(loss, self.full_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.full_model.trainable_variables))
        return loss

    def saveModel(self, name):
        if os.path.exists(os.path.join(f"Models/{name}.json")):
            shutil.copy(f"Models/{name}.json", f"Models/{name}.json.bak")
        if os.path.exists(os.path.join(f"Models/{name}.h5")):
            shutil.copy(f"Models/{name}.h5", f"Models/{name}.h5.bak")
        json = self.full_model.to_json()
        with open("Models/"+name+".json", "w") as json_file:
            json_file.write(json)

        self.full_model.save_weights("Models/"+name+".h5")

def generate_example():
    i = 0
    while True:
        yield tf.random.uniform((256,256,3))
        i +=1
        if i >= 100:
            break

def parse_example(example):
    features = {
        'ScenePath': tf.io.VarLenFeature(dtype=tf.string),
    }
    parsed_features = tf.io.parse_single_example(example, features=features)
    scene_path = parsed_features['ScenePath']
    frames = []
    frame_id = 0

    frame = tf.io.decode_jpeg()
    frames.append(image)

    return frames

def generate_scene_path(n):
    def generator():
        i = 0
        while i < n:
            if i in [338]:
                i += 1
                continue
            yield f"./scenes/scene-{i}"
            i += 1
    return generator


def parse_scene_example(scene_path):
    file_names = tf.data.Dataset.list_files(tf.strings.join([scene_path, tf.constant("/*.jpg", dtype=tf.string)]), shuffle=False)
    files_content = file_names.map(tf.io.read_file)
    frames = files_content.map(tf.io.decode_jpeg)
    frames = frames.map(lambda x: tf.image.convert_image_dtype(x, tf.float32))
    return frames

def prepare_sample_images(original_images, prediction_images):
    images = np.full((IMG_HEIGHT * 8, IMG_WIDTH * 8, 3), 255, dtype=np.uint8)

    original_images = original_images.numpy()
    prediction_images = prediction_images.numpy()

    for row in range(8):
        for col in range(4):
            images[row*IMG_HEIGHT:(row+1)*IMG_HEIGHT, 2*col*IMG_WIDTH:(2*col+1)*IMG_WIDTH] = original_images[row*4 + col]
            images[row*IMG_HEIGHT:(row+1)*IMG_HEIGHT, (2*col+1)*IMG_WIDTH:(2*col+2)*IMG_WIDTH] = prediction_images[row*4 + col]

    return images


def save_sample_image(model, testset, epoch_number):
    originals = next(iter(testset))

    original_frames, _ = tf.split(originals, 2, axis=1)
    original_frames = tf.squeeze(original_frames, axis=1)

    predictions = model.full_model(original_frames)

    # print(predictions[0])
    # tf.io.write_file(f"Results/test.jpg", tf.io.encode_jpeg(tf.saturate_cast(predictions[0], tf.uint8)))

    original_images = tf.image.convert_image_dtype(original_frames, tf.uint8, saturate=True)
    prediction_images = tf.image.convert_image_dtype(predictions, tf.uint8, saturate=True)

    images = prepare_sample_images(original_images, prediction_images)
    cv2.imwrite(f"Results/{epoch_number}.jpg", images)


if __name__ == '__main__':
    dp = DeepPassage()

    dataset = tf.data.Dataset.from_generator(generate_scene_path(906), (tf.string))
    dataset = dataset.map(parse_scene_example)

    # Split 90-10
    temp_dataset = dataset.enumerate()
    trainingset =  temp_dataset.filter(lambda i, data: i % 10 < 9)
    testset =  temp_dataset.filter(lambda i, data: i % 10 >= 9)

    del temp_dataset
    trainingset = trainingset.map(lambda i, data: data)
    trainingset = trainingset.map(lambda x: x.batch(2, drop_remainder=True))
    trainingset = trainingset.flat_map(lambda x: x.take(FRAME_PER_CLIP))
    trainingset = trainingset.batch(BATCH_SIZE, drop_remainder=True)
    trainingset = trainingset.shuffle(8192, reshuffle_each_iteration=True)

    testset = testset.map(lambda i, data: data)
    testset = testset.map(lambda x: x.batch(2, drop_remainder=True))
    testset = testset.flat_map(lambda x: x.take(FRAME_PER_CLIP))
    testset = testset.batch(BATCH_SIZE, drop_remainder=True)

    test_losses = []
    for i in tqdm(range(EPOCHS), unit='epochs'):
        test_losses.append(dp.train(trainingset, testset))
        np.save('test_losses.npy', np.array(test_losses))
        np.save('training_losses.npy', np.array(dp.losses))
        dp.saveModel('weights')
        save_sample_image(dp, testset.shuffle(1024), i)
