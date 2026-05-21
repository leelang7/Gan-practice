import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

from make_model import make_generator_model, make_discriminator_model

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_data():
    (X_train, y_train), (_, _) = fashion_mnist.load_data()

    X_train = (X_train - 127.5) / 127.5
    X_train = X_train[:, :, :, np.newaxis]
    y_train = to_categorical(y_train)

    return X_train, y_train

def generator_loss(base_loss, fake_output):
    # TODO: [지시사항 1번] 생성자의 손실 함수를 계산하세요.
    gen_loss = None

    return gen_loss

def discriminator_loss(base_loss, real_output, fake_output):
    # TODO: [지시사항 2번] 판별자의 손실 함수를 계산하세요.
    real_loss = None
    fake_loss = None
    disc_loss = real_loss + fake_loss

    return disc_loss

def train_step(images, generator, discriminator, gen_optimizer, disc_optimizer,
               base_loss, batch_size, noise_dim):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 생성자를 통해 랜덤 노이즈에서 새로운 이미지 생성
        generated_images = generator(noise, training=True)

        # 판별자를 통해 실제 이미지의 결과와 생성된 이미지의 결과를 계산
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # 생성자와 판별자의 손실 함수값 계산
        gen_loss = generator_loss(base_loss, fake_output)
        disc_loss = discriminator_loss(base_loss, real_output, fake_output)

    # 생성자와 판별자의 기울기 계산
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # 경사 하강법 적용
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 255.0, cmap='gray')
        plt.axis('off')

    plt.savefig('epoch_{:02d}.png'.format(epoch))
    elice_utils.send_image('epoch_{:02d}.png'.format(epoch))

def main():
    X_train, y_train = load_data()
    batch_size = 256
    dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(X_train.shape[0]).batch(batch_size)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    cross_entropy = BinaryCrossentropy(from_logits=True)
    gen_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)
    disc_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)

    epochs = 1
    noise_dim = 100
    initial_noise = tf.random.normal([16, noise_dim])

    for epoch in range(epochs):
        print("[Epoch {}]".format(epoch + 1))
        for image_batch in tqdm(dataset):
            train_step(image_batch, generator, discriminator, gen_optimizer, disc_optimizer,
                       cross_entropy, batch_size, noise_dim)

        generate_and_save_images(generator, epoch, initial_noise)

if __name__ == "__main__":
    main()