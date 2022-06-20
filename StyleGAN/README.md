# A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)

#### paper : https://arxiv.org/abs/1812.04948

※논문에서 자주 나오는 traditional model은 본 논문의 토대가 된 PGGAN을 의미한다. 따라서 기존 모델로 표현할 예정
※entangle = 이미지의 style이 얽혀있어 style controle등이 어렵기 때문에 저자는 disentanglement를 강조한다

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
- latent space Z를 8-layer MLP로 구성된 non-linear mapping network를 통해 W로 매핑 (disentangled)
- w가 AdaIN을 거쳐 generator에서 style을 control하는 역할을 한다
- conv마다 Noise를 추가로 넣어준다
- 그 외에는 기존 모델의 network를 토대로 한다.
### AdaIN?
$AdaIN(x_{i},y)=y_{s,i}\frac{x_{i}-\mu(x_i)}{\sigma(x_{i})}+y_{b,i}$

위 식의 분수 부분이 일반적인 정규화 부분이고 앞 뒤로 scaling과 bias를 적용하여 feature space의 statistics를 변경할 수 있게 한다.
이러한 Adaptive instance Normalization 방식을 generator에 추가하였다.

![image](https://user-images.githubusercontent.com/70709889/174528516-5fba3b3b-284f-49a5-b3a3-3202a2339d47.png)
**WGAN-GP**라는 loss를 사용하여 기존 모델(A)에서 한가지씩 추가하며 성능을 확인하였다.
```
※WGAN-GP란?
초창기 GAN모델의 불안정성을 개선하기 위한 Loss라고만 알아두자.
```
generated image를 잘 뽑아내기 위해 truncation trick을 사용하였는데 이는 뒤에서 결과 이미지를 뽑기 위해서만 사용했고 실제 train에는 적용하지 않았다고 후술한다.
### truncation trick
![image](https://user-images.githubusercontent.com/70709889/174530323-bff0c4e5-7348-45f3-930f-fbfa750df60b.png)

$\bar{w}=E_{z~P(z)}[f(z)]$
$w'=\bar{w}+\psi(w-\bar{w})$

매핑된 w 벡터를 그대로 사용하는 것이 아니라 $\overline{w}$만큼 떨어진 w'를 사용하는 것
## 3. Properties of the style-based generator
### Style mixing
각 style 간의 localize를 보장받기 위해 _mixing regularization_ 을 도입하였다.
- train시에 2개의 latent code $w_{1}, w_{2}$를 사용
- $w_{1}$을 적용하다가 crossover point이후 $w_{2}$를 적용한다
- 이 방식은 인접한 style끼리의 상관관계를 줄여준다
![image](https://user-images.githubusercontent.com/70709889/174530915-5e34a3cd-078f-4cff-9abb-8aa6d5c391d0.png)
- coarse style을 copying하였을 때 high-level의 style (포즈, 헤어스타일, 안경등)을 가져왔다
- middle style의 경우 얼굴 특징, 눈의 모양등 smaller한 변화를 확인할 수 있고 fine style의 경우 색상이나 배경등 미세한 detail이 변하였다
### Stochastic variation
