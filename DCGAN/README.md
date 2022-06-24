# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)
#### paper : https://arxiv.org/abs/1511.06434
## 0.Abstract
저자는 본 논문을 통해 unsupervised learning의 발전을 기대하는 것으로 보인다. CNN을 도입한 DCGAN을 소개할 것이며
generator와 discriminator가 이미지의 representations를 계층적으로 학습한다고 한다. => layer를 거치면서 생기는
high-level, low-level등의 feature를 의미하는건가?
## 1. Introduction
GAN을 학습시킴으로써 image representations를 추출하여 supervised learning을 위한 feature extractors로 활용할 수
있다고 기대한다. 하지만 GAN의 학습 불안정성과 가끔 무의미한 output을 내는 한계가 존재한다. 또한, GAN의
black box과 중간 layer의 representation을 이해하기 위한 연구도 필요하다고 한다. 이를 해결하기 위해
다음과 같은 방법을 제시한다.

- Convolution 구조를 도입하여 학습 안정화
- discriminator를 image classfier로 사용. => 침실 이미지를 학습했다면 합격 => 침실 이미지, 불합격 => 침실 이미지가 
아니다. 이런 느낌인가.. bedroom, diniing room같은 label로 판별하는게 아니라 data distribution으로 판별하니까 unsupervised 라고 표현한 듯.
- GAN이 학습한 filter를 visualize하고 각 filter가 특정 object를 drawing 한다는 것을 보여주겠다
- generator가 semantic한 vector 연산이 가능하다는 것을 보여주겠다. ex) KING - MAN + WOMAN = QUEEN
## 2. Related Work
### 2-1. Representation Learning from Unlabeled Data
기존에는 unsupervised representation learning에 cluster 또는 auto encoder를 사용했다. 이러한 방법들은
pixel 단위로 feature를 학습하기 좋은 방법들이다.
### 2-2. Generating Natural Images
Generative model을 두가지 카테고리가 있다.
- non-parametic
  - 종종 database image 또는 image patches와 matching하며 texture synthesis, spuer-resolution, in-paiting에 사용되었다.
- parametic
  - 많은 발전이 있었고 variational sampling 방식은 blurry 하다는 문제점이 있다. (당시 기준)
             향후 공부해봐야겠다. 
  - diffusion model은 noisy와 이해할 수 없는 이미지가 생성된다는 문제점이 있다. (당시 기준)
  - laplacian pyramid를 사용한 접근법도 있었으나 여러 모델을 결합하는 과정에서 noise가 발생한다.
  - recurrent방식과 deconvolution network 방식으로 natural image를 생성하는데 성공하였지만 generator를 supervised task에 적용하지는 못하였다.
### 2-3. Visualizing the internals of CNNs
뉴럴넷의 지속적인 비판 중 하나는 black-box에 관한 것이다. CNN은 deconvolution과 maximal actiation으로 network에서 특정 filter를 찾을 수 있고 이처럼 input에 gradient descent를 적용함으로써 filters의 부분집합으로 이루어진 image를 얻을 수 있다.
## 3. Approach and Model Architecture
![image](https://user-images.githubusercontent.com/70709889/175698934-9a4c633e-68e0-47b6-86d6-4985a6e4cfe0.png)

다음과 같은 Architecture guideline을 제시한다.
- pooling layer를 strided conv(Discriminator), fractional-strided conv(Generator)로 대체한다
  - network가 고유의 spatial한 downsampling을 학습할 수 있도록 해준다
- Generator와 Discriminator 모두 Batchnorm을 사용한다
  - input을 zero mean과 unit variance를 가지도록 정규화하여 학습을 안정화해준다
  - 이는 초기화 문제를 해결해주고 더 깊은 model에서 gradient가 잘 흐를 수 있게 해준다
  - deep generator가 collapse 문제를 예방하게 해준다는 것을 증명해준다
  - 모든 layer에 적용하는 것은 sample oscillation과 모델 불안정성을 야기하기 때문에 generator output layer과 discriminator input layer에는 적용하지 않았다. 왜 output과 input인지? 실험적인 결과인가?
- hidden layer에서 FC-layer를 제거한다
  - FC-layer를 제거하는 것이 최신 트렌드이며 GAP는 모델 안정화에는 도움이 되지만 수렴 속도에는 좋지 않았음
  - highest conv feature를 Generator와 Discriminator에 각각 input과 output으로 연결하는 것이 성능이 좋았다
  - uniform noise Z를 입력으로 받는 GAN의 첫번째 layer는 행렬곱을 한다는 면에서 fully connected라고 불릴 수 있지만 4차원 tensor로 reshape 해주는 역할을 하고 convolution stack의 시작이 된다
- Generator에서 ReLU 함수를 사용하고 output layer에는 Tanh를 사용한다
  - model이 saturate하고 trainig distributino의 color space를 cover하는 것을 더 빠르게 해준다
- Discriminator는 모든 layer에서 LeakyReLU를 사용한다
  - higher resolution에 도움이 된다
## 4. Details of Adversial Training 
Large-scale Scene Understanding (LSUN), Imagenet-1k, Faces dataset으로 학습하였다.
### 4-1. LSUN
생성된 이미지 퀄리티가 증가함에 따라 over-fitting / memorization의 우려가 발생하였다. 즉, training sample을 치팅했는지 아닌지 증명하기 위해 one epoch 결과를 제시하였다.
![image](https://user-images.githubusercontent.com/70709889/175699225-1dc584b9-67b5-43e0-b0b1-dd28b7b132d4.png)

online learning을 모방하여 1 epoch만 학습했기 때문에 over-fitting 혹은 memorization의 결과가 아님을 보여준다.

memorizing에 대한 확률을 더 줄이기 위해 중복 이미지를 제거하였다.
1. 3027-128-3072 de-noising dropout regularized RELU auto-encoder를 32x32로 downsampling된 training sample의 center-crop에 맞춘다
2. 그 결과인 code layer activation은 RELU threshold에 의해 이진화된다
3. 이는 semantic-hashing 하기 용이한 형태이고 linear time 중복 제거를 수행한다
4. 충돌하는 hash들을 확인한 결과 높은 precision을 보였고 결과적으로 275000개의 중복 이미지를 제거하였다
### 4-2. Faces
사람 얼굴 이미지에서 고해상도를 유지하면서 OpenCV face detector를 통해 350000개의 face boxes를 수집하였다. No augmentation.
### 4-3. Imagenet-1k
unsupervised training을 위해 사용하였고 augmentation은 적용하지 않았다.
## 5. Empirical validation of DCGANs Capabilities
DCGAN을 다양하게 활용한 방안들
### 5-1. Using GAN as Feature Extractor
unsupervised representation의 quality를 평가할 때 feature extractor로 사용한 뒤 linear model에서의 성능을 확인한다. 이를 위해 discriminator 모든 layer의 conv feature를 각각 maxpooling을 통해 4x4 spatial grid로 생성한 뒤, flatten과 concat을 통해 linear L2-SVM으로 분류했다.
![image](https://user-images.githubusercontent.com/70709889/175719357-bca9ec81-8ade-4e6e-88a3-c2af598e7cf1.png)

다양한 pipeline과 비교한 결과 K-means보다는 우세하지만 Exemplar CNN보다는 약한 성능을 보인다. 저자들은 이에 대해 finetunning으로 개선 여지를 남긴다.
### 5-2. Classifying SVHN Digits using GANS as as Feature Extracor
![image](https://user-images.githubusercontent.com/70709889/175719808-9c171402-e632-40c4-b43c-d09f1a072cab.png)

labeled data가 부족한 task에 대해 discriminator의 feature를 사용하였으며 다른 CNN 변형 모델들보다 좋은 성능을 보인다. 여러 실험 결과 DCGAN에 사용된 CNN 구조가 높은 성능의 key가 아님을 발견했다.
## 6. Investigating and Visualizing the Internals of the Networks
본 논문에서 가장 재밌는 파트
### 6-1. Walking in the latent space
latent space에서 움직일 때 급격한 변화를 보인다면 memorization의 증거라고 볼 수 있다. 만약 semantic한 변화가 발생한다면, model이 relevant하고 interesting한 representation들을 학습했다는 의미이다.
```
"we can reason that the model has learned relevant and interesting representations."
뭔소리여? relevant는 entangled를 말하는건가?
```
![image](https://user-images.githubusercontent.com/70709889/175720227-985204fc-ab85-4e86-97ef-51eadd53683b.png)

보다시피 부드러운 변화를 보이며 10th row에서는 TV가 window로 바뀌는 등의 semantic change가 발생한다.
### 6-2. Visualizing the Discriminator Features
앞서 말했듯, CNN의 black box가 가장 큰 문제 중 하나이기 때문에 guided backpropagion을 사용해 discriminator가 학습한 feature들을 시각화하였다.
![image](https://user-images.githubusercontent.com/70709889/175720653-bab1ceea-df67-4117-8aa0-37f39534f48c.png)

좌측의 random fileter와 비교하여 우측의 feature에서는 침대나 창문같은 침실의 특징을 학습했다는 것을 알 수 있다.
### 6-3. Manipulating the Generator Representation
#### 6-3-1. Forgetting to Draw Certain Objects
6-2와 연관하여 그렇다면 generator가 학습한 representation에 대해 의문을 가져보자. samples에는 침대, 창문, 램프, 문등 여러가서 component가 있고 이 중에서 generator가 창문을 제거한 representation을 생성하도록 하였다. Second highest conv layer의 feature에 logistic regression을 적용하여 창문에 대해 feature activation이 발생하는지 예측한다. feature map의 가중치가 0보다 크다면 창문에 관여한다는 뜻이므로 모든 spatial location에서 제거하였다.
![image](https://user-images.githubusercontent.com/70709889/175721186-7fb243dd-f198-400c-8520-18571393807b.png)

상단 row는 수정하지 않은 samples이고 그에 반해, 하단 row에서 창문이 사라진 samples를 확인할 수 있다. 잘 안 보이지만 창문이 사라진 자리를 network가 다른 object로 대체하려고 했다는데 이 부분이 더 신기한 듯...
#### 6-3-2. Vector Arithmetic on Face Samples
![image](https://user-images.githubusercontent.com/70709889/175721476-901bcad4-cdae-48b3-b085-32da8f1f2439.png)

vector 산술연산을 통해 semantic한 표현이 가능하다. 각 단어에 대해 3개의 samples를 평균한 Z vector에 대해서 안정적인 생성을 한다.
![image](https://user-images.githubusercontent.com/70709889/175722225-af2b6090-693e-4f74-b745-6ac22370fdf4.png)

위와 같이 'turn' vector를 통해 object를 transform 할 수 있다. 이를 interpolation 한다면 다양한 각도의 얼굴을 생성하는 것도 가능하며 이는 더 많은 연구가 진행된다면 향후 conditional generative model에 필요한 data의 양을 줄여줄 수 있을 것이라고 암시한다.
## 7. Conclusion and Future Work
- 안정적인 구조 제시
- GAN이 생성한 representation을 통해 supervised learning에 적용 가능
- 모델이 오래 학습하면 collapse 하는 등 불안정성이 남아있다
- 이러한 불안정성, latent space에 대한 연구 기대
- framework를 video, audio에 확장할 수 있을 것이라고 기대

## 8. Supplementary Material
### 8-1. Evaluating DCGANs Capability to Capture Data Distribution
DCGAN의 conditional version 성능을 확인하기 위해 기존 분류 metric으로 평가하였다. 이 과정에서 nearset neighbor classifier를 사용하였고 batchnorm으로 생성된 noise가 data distribution을 모사하는데 도움이 되는 것 같다고 추측한다. 비교 결과는 다음과 같다.
![image](https://user-images.githubusercontent.com/70709889/175743460-56d200b5-ee65-498a-aaa5-153c14a5bb5b.png)

Real Data보다 분류가 잘 되는 것 같은데 신기하다...
