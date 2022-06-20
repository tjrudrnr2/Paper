# A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)

#### paper : https://arxiv.org/abs/1812.04948

※논문에서 자주 나오는 traditional model은 본 논문의 토대가 된 PGGAN을 의미한다. 따라서 기존 모델로 표현할 예정

## Abstract & Introduction
- 기존의 PGGAN (style transfer literature)에서 몇가지를 개선하였다
  - high-level attribute 분류
  - 이미지 생성에 있어서 stochastic variation
  - intuitive, scale-specific 한 control의 가능
- 기존의 distribution quality metric에서의 성능 향상
- 높은 성능의 interpolation, disentangles
- interpolation, disentangles를 정량화하기 위한 두 가지 method 제시
- 고퀄리티의 dataset (FFHQ) 공개
## 1. Introduction
최근까지도 GAN은 black box로써 동작하기 때문에 image synthesis process와 latent space 등에 대한 이해가 부족하고 서로 다른 GAN 끼리 정량적으로 비교할 metric이 마땅하지 않다.
### StyleGAN의 가장 큰 특징
- 기존 모델의 discriminator를 그대로 사용
- latent space가 disentangled 하도록 조정
- constant input을 중간 layer에 삽입
- 각 conv layer에서 이미지의 'style'을 조정 가능
## 2. Style-based generator
![image](https://user-images.githubusercontent.com/70709889/174525865-d10bc42d-a6ae-4ff6-af8d-910f4b3e19b2.png)
- latent space Z를 non-linear mapping network를 통해 W로 매핑
- 
