# A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)

#### paper : https://arxiv.org/abs/1812.04948

※논문에서 자주 나오는 traditional model은 본 논문의 토대가 된 PGGAN을 의미한다. 따라서 앞으로 기존 모델로 표기할 예정

※entangle = 이미지의 style이 얽혀있어 style 개별의 control등이 어렵기 때문에 저자는 disentanglement를 강조한다

## Abstract & Introduction
- 기존의 PGGAN (style transfer literature)에서 몇가지를 개선하였다
  - high-level attribute 분리
  - stochastic variation ex) 머리 휘날리는 정도, 주름, 피부 모공
  - intuitive, scale-specific 한 control의 가능
- 기존의 distribution quality metric에서의 성능 향상
- 높은 성능의 interpolation, disentangles
- interpolation, disentangles를 정량화하기 위한 두 가지 method 제시
- 고퀄리티의 dataset (FFHQ) 공개
## 1. Introduction
최근까지도 GAN은 image synthesis process와 latent space 등에 대한 이해가 부족하고 서로 다른 GAN 끼리 정량적으로 비교할 metric이 마땅하지 않다.
### StyleGAN의 가장 큰 특징
- 기존 모델의 discriminator와 loss fucntion을 그대로 사용
- latent space가 disentangled 하도록 조정
- disentanglement를 측정할 2가지 methond => more linear, less entangled representation
- constant input을 중간 layer에 삽입
- 각 conv layer에서 이미지의 'style'을 조정 가능
## 2. Style-based generator
![image](https://user-images.githubusercontent.com/70709889/174525865-d10bc42d-a6ae-4ff6-af8d-910f4b3e19b2.png)
- 기존에는 latent cod z가 input layer에 그대로 삽입
- latent space Z를 8-layer MLP로 구성된 non-linear mapping network를 통해 W로 매핑 (disentangled)
- w가 affine transform을 거쳐 style로 변환
- style이 AdaIN을 통해 generator에서 style을 control하는 역할을 한다
- conv마다 Noise를 추가로 넣어준다 => stochastic variance
- 마지막 layer에서 1x1 conv로 RGB로 변환
- 4x4 부터 1024x1024까지 upsampling 하는 과정 => 9개 block, 18 layer
### Adaptive Instance Normalization (AdaIN)?
$AdaIN(x_{i},y)=y_{s,i}\frac{x_{i}-\mu(x_i)}{\sigma(x_{i})}+y_{b,i}$

- Instance 별로 normalization
- 위 식의 분수 부분이 일반적인 정규화 부분이고 앞 뒤로 scaling과 bias를 적용하여 feature space의 statistics를 변경할 수 있게 한다.
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

분포의 low density에서 이미지를 생성하는 것을 피하기 위해서 매핑된 w 벡터를 그대로 사용하는 것이 아니라 $\overline{w}$만큼 떨어진 w'를 사용하는 것
## 3. Properties of the style-based generator
- styles의 집합으로부터 새로운 이미지를 생성하는 과정이다
### Style mixing
각 style 간의 localize를 보장받기 위해 _mixing regularization_ 을 도입하였다.
- train시에 2개의 latent code $w_{1}, w_{2}$를 사용
- $w_{1}$을 적용하다가 crossover point 이후 $w_{2}$를 적용한다
- 이 방식은 인접한 style끼리의 상관관계를 줄여준다
![image](https://user-images.githubusercontent.com/70709889/174530915-5e34a3cd-078f-4cff-9abb-8aa6d5c391d0.png)
- coarse style을 copying하였을 때 high-level의 style (포즈, 헤어스타일, 안경등)을 가져왔다
- middle style의 경우 얼굴 특징, 눈의 모양등 smaller한 변화를 확인할 수 있고 fine style의 경우 색상이나 배경등 미세한 detail이 변하였다
### Stochastic variation
![image](https://user-images.githubusercontent.com/70709889/174796693-428dcedb-63b0-40f0-96f4-192a17208bf2.png)

- 각 conv마다 noise를 추가함으로써 pixel별로 noise가 적용되게 하였다
- high-level attributes는 그대로 두고 확률적인 부분에만 영향을 미친다

![image](https://user-images.githubusercontent.com/70709889/174795865-6547daa5-3beb-48e9-91d2-9ee821856b2f.png)

(a) Noise 모든 layer에 적용
(b) Noise 적용 X
(c) 뒷 layer에만 noise 적용
(d) 앞 layer에만 noise 적용
## Disentanglemnet studies
![image](https://user-images.githubusercontent.com/70709889/174791341-c960bca5-f305-430a-b8b5-28a9a59eb94a.png)

- 기존에 Z를 input으로 넣을 때는 subspace가 non-linear하다 => entangled
  ∵ Z는 training data의 분포를 따른다
- W로 매핑한 경우restrict가 없다 => more linear
- entangled 보다 disentangled representation에서 이미지를 생성하기 쉬울 것으로 예상
==> 기존 metric은 input으로 들어가는 latent code가 필요하기 때문에 새로운 방법 제시
### 1. Perceptual path length
- 두 latent-space를 interpolation 할 때 이미지가 비선형적으로 변한다 ex) 양 사이드에 없는 faeture가 middle에 등장
=> entangled의 근거. latent space를 interpolation 했을 때 얼마나 급격하게 변하는지 보자!
```
VGG16으로 perceptual length를 계산
```
#### Z space
![image](https://user-images.githubusercontent.com/70709889/174802361-bb9d6271-83c9-44b3-bbb8-85cbcd220c69.png)

#### W space
![image](https://user-images.githubusercontent.com/70709889/174801809-8bb69ff6-18b1-4f96-91fe-70d997782c1c.png)

- slerp = 구면체 보간법
- lerp = 선형 보간법
### 2. Linear separability
- latent space가 충분히 disentangle 되었다면 특정 factor에 해당하는 vector를 찾을 수 있을 것
  => 얼마나 선형적으로 잘 분리되는지 측정하자
1. 이미지 생성
2. classifier를 통해 분류. 저자들은 discriminator와 같은 구조 사용
3. confidence로 정렬한 후 상위 절반만 취급
4. conditional entropy ($H(Y|X)$)를 구한다.
    X = linear SVM을 통해 latent space를 label 별로 분류한 classes
    Y = classifier를 통해 분류된 classes
5. i 개의 attributes에 대해 $exp(\Sigma_{i}H(Y_{i}|X_{i}))$
 
    => 본래 latent space보다 몇 개의 label이 모자란 지
![image](https://user-images.githubusercontent.com/70709889/174841349-9570cef0-a127-40af-893c-b288f6a0f8a3.png)

![image](https://user-images.githubusercontent.com/70709889/174841439-82e249b2-e4cc-46b3-b4a7-b345085ad61e.png)
## 5. Conclusion
- 기존의 PGGAN보다 성능이 좋다
- high-level attribute의 분리 (개별 style의 control)
- stochastic effect
- latent space의 선형화
    => GAN의 black box에 대한 의미있는 연구
    
- perceptual path length, linear separabiliry metric 제시
