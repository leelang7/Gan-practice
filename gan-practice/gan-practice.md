# 1. **Style Transfer (1)**

이번 실습에서는 Style Transfer를 수행하기 위한 첫번째 단계로 Style Transfer 모델과 Style Transfer에 사용할 CNN의 Layer를 선택하도록 하겠습니다.

![img](https://cdn-api.elice.io/api-attachment/attachment/1d0b5d3e64ae40f3a0e7293b7a3ab707/style_transfer.png)

Style Transfer를 위해 사용할 기반 CNN 모델은 VGG-19 모델입니다. 이 VGG-19 모델 내에서 style loss를 계산하기 위한 5개의 Layer와 content loss를 계산하기 위한 1개의 Layer를 고를 것입니다.

이후 Style Transfer를 위해 새로 생성될 이미지가 학습할 스타일을 가지는 **style targets**과 학습할 원래 이미지(컨텐츠 이미지)에 해당하는 **content targets** 까지 생성하도록 하겠습니다.

## 지시사항

Style Transfer를 수행하기 위한 모델은 `StyleTransferModel`이라는 클래스로 구현되어 있습니다.

지시사항에 따라 코드를 채워넣어 클래스 구현을 완성하세요.

1. `StyleTransferModel`의 `call` 메소드 내에서 주어진 `inputs`으로 style feature map들을 뽑아낸 다음에 각 feature map의 Gram matrix를 구하세요.
2. `StyleTransferModel`의 `call` 메소드 내에서 주어진 `inputs`으로 content feature map들을 뽑아내세요.





# 2. **Style Transfer (2)**

이번 실습에서는 Style Transfer의 두번째 단계로 **style loss**와 **content loss**를 정의하고 모델을 직접 학습시켜 결과를 확인해보도록 하겠습니다.

Style Transfer 모델에서 style loss와 content loss는 모두 새로 생성될 이미지와 style 또는 content 이미지의 feature map을 통해 계산합니다. 즉 새로 생성될 이미지의 feature map 행렬을 Fl*F**l*, style 또는 content 이미지의 feature map 행렬을 Pl*P**l* 이라고 한다면 다음과 같은 형태로 loss를 계산합니다.

![image-20260521112952309](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20260521112952309.png)

Style 이미지의 경우엔 Fl*F**l*과 Pl*P**l*이 Gram 행렬을 의미한다는 것만 다를 뿐 본질적으로는 두 행렬의 각 원소값을 뺀 것을 제곱하여 모두 더하는 것입니다.

## 지시사항

1. Style loss를 계산하는 함수 `compute_style_loss`를 완성하세요. 새로 생성될 이미지와 style 이미지를 통해 각 Layer 별로 계산된 Gram 행렬은 각각 `output`과 `target`으로 주어져 있습니다.
2. Content loss를 계산하는 함수 `compute_content_loss`를 완성하세요. 새로 생성될 이미지와 content 이미지를 통해 각 Layer 별로 계산된 feature map은 각각 `output`과 `target`으로 주어져 있습니다.
3. 전체 loss를 계산하는 함수 `compute_total_loss`를 완성하세요. 전체 loss는 style loss를 `style_weight`만큼 곱하고 content loss를 `content_weight`만큼 곱한 것을 더해서 계산합니다.



# 3. **Generative Adversarial Net (1)**

이번 실습에서는 GAN 모델을 학습하기 위한 첫번째 단계로 **생성자(Generator)** 와 **판별자(Discriminator)** 모델을 정의하도록 하겠습니다.

![img](https://cdn-api.elice.io/api-attachment/attachment/a80949c4149f4f8f85cfe3b89856ce88/gan.png)

생성자 모델은 일차원의 랜덤 노이즈 벡터를 통해 학습 데이터셋의 이미지와 비슷한 이미지를 생성하는 것이 목표입니다. 따라서 이 모델에서는 Convolution 연산을 통해 이미지의 사이즈를 늘려가는 Layer인 **Transposed Convolutional Layer**를 사용합니다.

판별자 모델은 입력 받은 이미지가 실제 이미지인지 생성된 이미지인지를 판단하므로 일반적인 CNN의 구조와 동일합니다.

## 지시사항

1. 생성자 모델을 만드는 함수 `make_generator_model`를 완성하세요. Layer 구성은 다음과 같습니다.
   - `layers.Dense`
     - 노드 개수: `7` ×× `7` ×× `256`
     - `use_bias=False`
     - `input_shape=(100,)`
   - `layers.BatchNormalization`
   - `layers.LeakyReLU`
   - `layers.Reshape`
     - 변경할 이미지 모양: `(7, 7, 256)`
   - `layers.Conv2DTranspose`
     - 커널 개수: `128개`
     - 커널 크기: `(5, 5)`
     - Stride: `(1, 1)`
     - Padding: `"same"`
     - `use_bias=False`
   - `layers.BatchNormalization`
   - `layers.LeakyReLU`
   - `layers.Conv2DTranspose`
     - 커널 개수: `64개`
     - 커널 크기: `(5, 5)`
     - Stride: `(2, 2)`
     - Padding: `"same"`
     - `use_bias=False`
   - `layers.BatchNormalization`
   - `layers.LeakyReLU`
   - `layers.Conv2DTranspose`
     - 커널 개수: `1개`
     - 커널 크기: `(5, 5)`
     - Stride: `(2, 2)`
     - Padding: `"same"`
     - `use_bias=False`
     - 활성화 함수: `tanh`

1. 판별자 모델을 만드는 함수 `make_discriminator_model`을 완성하세요. Layer 구성은 다음과 같습니다.
   - `layers.Conv2D`
     - 커널 개수: `64개`
     - 커널 크기: `(5, 5)`
     - Stride: `(2, 2)`
     - Padding: `"same"`
     - `input_shape=[28, 28, 1]`
   - `layers.LeakyReLU`
   - `layers.Dropout`
     - 비율: `0.3`
   - `layers.Conv2D`
     - 커널 개수: `128개`
     - 커널 크기: `(5, 5)`
     - Stride: `(2, 2)`
     - Padding: `"same"`
   - `layers.LeakyReLU`
   - `layers.Dropout`
     - 비율: `0.3`
   - `layers.Flatten`
   - `layers.Dense`
     - 노드 개수: `1개`



# 4. Generative Adversarial Net (2)

이번에는 앞서 생성한 두 모델을 Fashion MNIST 데이터셋으로 학습하는 과정을 거칠 것입니다.

GAN에서는 생성자와 판별자 모델을 학습하기 위한 손실 함수가 따로 필요합니다. 그리고 나서 최종 손실 함수는 아래와 같이 둘을 더하는 것으로 계산합니다.

![image-20260521112919913](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20260521112919913.png)

이번 실습에서는 이 두 손실 함수로 모델을 학습하는 과정을 코드를 통해 알아보도록 하겠습니다.



> 모델 학습 과정에서 다소 시간이 소요될 수 있는 코드를 포함하고 있습니다.

## 지시사항

모든 코드를 완성하게 되면 모델 학습에 오랜 시간이 걸리게 됩니다. 하나의 epoch을 완료하는데 대략 6~7분 가량의 시간이 소요되므로 **기본 epoch은 1로 설정**되어 있습니다. 유의미한 결과를 확인해보고 싶다면 구글 colab 이나 로컬의 GPU를 활용하여 30 epoch 이상 학습해보시길 권장합니다. 엘리스 플랫폼 내 제한시간은 **30분**입니다.

지시사항에 따라 코드를 완성하세요.

1. 생성자 모델을 위한 손실 함수를 계산하는 함수 `generator_loss`를 완성하세요.
   - 사용할 기반 손실 함수는 `base_loss`로 주어집니다.
   - 이 경우 실제 타겟은 `fake_output`과 **같은 모양**을 가지면서 **모든 값이 1**인 텐서입니다. `tf.ones_like` 함수를 참고하세요.
2. 판별자 모델을 위한 손실 함수를 계산하는 함수 `discriminator_loss`를 완성하세요.
   - 마찬가지로 기반 손실 함수는 `base_loss`로 주어집니다.
   - 실제 이미지와 비교될 타겟은 `real_output`과 **같은 모양**을 가지면서 **모든 값이 1**인 텐서입니다.
   - 생성된 가짜 이미지와 비교될 타겟은 `fake_output`과 **같은 모양**을 가지면서 모든 값이 0인 텐서입니다. `tf.zeros_like` 함수를 참고하세요.

### Tips

실제 하나의 배치를 통해 모델 학습을 수행하는 함수는 `train_step`에 구현되어 있습니다. 생성자와 판별자에 이미지를 넣고 손실 함수를 계산하여 경사 하강법이 이루어지는 과정이 어떤지 확인해보세요. 이 부분이 **GAN 모델 학습의 핵심**입니다.