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
  - 
