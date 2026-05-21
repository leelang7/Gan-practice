import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
from tensorflow.keras import layers, Sequential

def make_generator_model():
    model = Sequential()

    # TODO: [지시사항 1번] 생성자 모델을 만드세요.
    model.add(None)
    
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    
    # TODO: [지시사항 2번] 판별자 모델을 만드세요.
    model.add(None)

    return model
    
def main():
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    
    generator.summary()
    discriminator.summary()

if __name__ == "__main__":
    main()