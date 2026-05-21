import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

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
    
class StyleTransferModel(Model):
    def __init__(self, style_layers, content_layers):
        super(StyleTransferModel, self).__init__()
        
        self.vgg_model = VGG19(include_top=False, weights='imagenet')
        self.vgg_model.trainable = False

        self.style_layers = style_layers
        self.content_layers = content_layers

        self.style_extractor = self.get_extractor(style_layers)
        self.content_extractor = self.get_extractor(content_layers)

    # Style feature map과 content feature map을 뽑아내기 위한
    # extractor를 정의합니다.
    def get_extractor(self, layer_names):
        outputs = [self.vgg_model.get_layer(name).output for name in layer_names]
        model = Model([self.vgg_model.input], outputs)

        return model

    # Feature map에서 Gram 행렬을 계산합니다.
    def gram_matrix(self, input_tensor):
        result = tf.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    
        return result / num_locations

    def call(self, inputs):
        inputs = preprocess_input(inputs)
        
        # TODO: [지시사항 1번] 주어진 입력 이미지에서 style feature map을 뽑아내고
        # 이를 통해 gram matrix를 구하세요.
        style_outputs = self.style_extractor(inputs)
        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]
        
        # TODO: [지시사항 2번] 주어진 입력 이미지에서 content feature map을 뽑아내세요.
        content_output = self.content_extractor(inputs)

        content_dict = {content_name: value 
                        for content_name, value 
                        in zip(self.content_layers, content_output)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}
    
        return {"content": content_dict, "style": style_dict}
        
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
    
    print("content targets:", list(content_targets.keys()))
    print("style targets:", list(style_targets.keys()))
    print("두 종류의 target을 생성하였습니다.")
        
if __name__ == "__main__":
    main()