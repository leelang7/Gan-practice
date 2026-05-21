import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image

from st_model import StyleTransferModel

def load_img(path_to_img):
    # 이미지의 가로 또는 세로의 최대 픽셀 수를 512개로 제한합니다.
    max_dim = 512

    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # 원본 이미지에서 가장 픽셀이 많은 쪽의 픽셀수를 가져와
    # 512로 줄였을 때의 비율만큼 이미지 크기를 조정합니다.
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)

    # batch에 해당하는 차원을 추가해줍니다.
    img = img[tf.newaxis, :]

    return img
    
def compute_style_loss(style_outputs, style_targets):
    style_losses = []
    for name in style_outputs:
        output_shape = style_outputs[name].shape[1:]
        output = style_outputs[name]
        target = style_targets[name]

        # TODO: [지시사항 1번] output과 target 사이의 loss를 계산하기 위해
        # 행렬의 원소값을 뺀 것을 제곱하여 모두 더하세요.
        layer_loss = None
        layer_loss /= 4

        style_losses.append(layer_loss)

    style_loss = tf.reduce_sum(style_losses) / len(style_losses)

    return style_loss

def compute_content_loss(content_outputs, content_targets):
    assert len(content_outputs) == 1

    name = list(content_outputs.keys())[0]
    output = content_outputs[name]
    target = content_targets[name]
    
    # TODO: [지시사항 2번] output과 target 사이의 loss를 계산하기 위해
    # 행렬의 원소값을 뺀 것을 제곱하여 모두 더하세요.
    content_loss = None
    content_loss /= 2

    return content_loss
    
def compute_total_loss(style_loss, style_weight, content_loss, content_weight):
    # TODO: [지시사항 3번] 전체 loss를 계산하세요.
    total_loss = None
    
    return total_loss

def train_step(style_transfer_model, optimizer, generated_image,
               style_targets, style_weight, content_targets, content_weight):
    with tf.GradientTape() as tape:
        outputs = style_transfer_model(generated_image)
        style_outputs = outputs["style"]
        content_outputs = outputs["content"]

        style_loss = compute_style_loss(style_outputs, style_targets)
        content_loss = compute_content_loss(content_outputs, content_targets)
        
        loss = compute_total_loss(style_loss, style_weight, content_loss, content_weight)
        print("loss: {}".format(loss))

        grad = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    
    return Image.fromarray(tensor)
    
def main():
    content_image = load_img("./content_image.jpg")
    style_image = load_img("./style_image.jpg")

    # style feature map과 content feature map을 뽑아내기 위한
    # VGG-19 내의 Layer를 선택합니다.
    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    # Style Transfer 모델을 불러옵니다.
    style_transfer_model = StyleTransferModel(style_layers, content_layers)
    style_targets = style_transfer_model(style_image)["style"]
    content_targets = style_transfer_model(content_image)["content"]

    # 새로 생성될 이미지입니다. 학습 가능하도록 Variable로 정의합니다.
    generated_image = tf.Variable(content_image)

    # Style loss와 content loss에 적용할 weight입니다.
    style_weight = 100
    content_weight = 1
    
    # Optimizer는 Adam을 사용합니다.
    optimizer = Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # 새로 생성될 이미지를 학습합니다.
    for i in range(10):
        print("[Epoch {}] ".format(i), end="")
        train_step(style_transfer_model, optimizer, generated_image,
                   style_targets, style_weight, content_targets, content_weight)

    tensor_to_image(generated_image).save("style_transfer.png")
    elice_utils.send_image("style_transfer.png")
    
if __name__ == "__main__":
    main()