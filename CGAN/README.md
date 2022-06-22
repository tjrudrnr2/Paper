# Conditional Generative Adversarial Nets (CGAN)

#### Paper : https://arxiv.org/abs/1411.1784

## 0. Abstract
기존 GAN에 data y를 추가로 feed하여 Generator와 Discriminator에 condition을 부여하는 것이 목표이다. class label을 조건부로 하는 MNIST 데이터를 생성할 수 있고 추가로 multi-modal에서도 사용될 수 있다는 것을 보여준다. 이 논문에서는 image tagging을 예시로 제시하였다.
## 1. Introduction
기존 unconditioned model은 생성될 이미지의 모드(label)을 control할 수 없다는 단점이 있다. 하지만 conditional 정보를 추가로 입력하여 data 생성 과정에 직접적으로 관여할 수 있다. 이러한 conditionla 정보는 주로 class label이며 다른 modality의 data로도 가능하다. ex) image를 설명하는 단어
## 2. Related Work
### Multi-modal Learning for Image Labeling
supervised network가 많은 발전을 이루었지만 극도로 많은 수의 output category를 수용하거나 많은 문제들이 1:다 매핑으로 이루어짐에도 불구하고 대부분 1:1 매핑에 focus한다는 점이다.
ex) 이미지를 annotation 하는 사람마다 다른 언어로 나타낼 수 있음

이러한 문제를 해결하기 위해
1. 다른 modality로부터 추가 정보를 가져오는 것
2. conditional generative model을 사용하는 것

## 3. Conditional Adversial Nets
### 3-1. Generative Adversial Nets
[GAN](https://github.com/tjrudrnr2/Paper/tree/main/GAN)에 관한 설명은 생략
### 3-2. Conditional Adversial Nets
- Generator와 Discriminator에 조건부 정보 y input layer를 추가함으로써 conditional을 나타낼 수 있다.
$\underset{G}{min}\underset{D}{max}V(D,G)=E_{x->p_{data}(x)}[log D(x|y)]+E_{z->p_z(z))]}[log(1-D(G(z|y)))]$
- 위의 수식을 보면 기존 GAN의 loss function에서 $D(x) \rightarrow D(x|y), G(z) \rightarrow G(z|y)$로 조건부 변수 y가 추가된 것을 알 수 있다.

![image](https://user-images.githubusercontent.com/70709889/175016788-26b4a7b4-9425-46ff-896c-6e4743ef3245.png)
## 4. Experimental Results
### 4-1. Unimodal
- MNIST class label을 one-hot 벡터로 인코딩하여 주입한다
- Generator에서 noise prior z와 y 모두 ReLU와 함께 hidden layer에 매핑되며 마지막에는 sigmoid를 사용하였다
- Discriminator에서는 x와 y를 각각 maxout layer에 제공한 뒤 maxout layer에 한번 더 joint 해준 후 sigmoid layer를 거친다

![image](https://user-images.githubusercontent.com/70709889/175017808-bc2580d4-2d91-4d54-8b35-510882ad7e3a.png)

Parzen window 방식으로 log-likelihood를 측정했을 때 결과는 좋지 않았으며 저자들은 성능보다는 개념의 증명을 강조한다. 또한, 하이퍼 파라미터 튜닝과 구조에 대한 탐구를 한다면 성능을 올릴 수 있을 것이라고 한다
### 4-2. Multimodal
- 이미지 features를 conditionla으로 입력받아 tag-vectors를 생성하도록 하였다
- Flickr 이미지와 그에 관한 UGM(user-generated metadata)를 데이터셋으로 사용하였다
- conv model과 language model로 image와 tag feature를 추출하였다
- 생성된 이미지 tag 중 cosine similarity가 비슷한 단어들을 추린 결과는 아래와 같다.
![image](https://user-images.githubusercontent.com/70709889/175021112-d352d2b8-b667-4d88-887a-e747e8e52c9d.png)
## 5. Future Work
본 논문의 결과는 preliminary하며 CGAN의 잠재력을 보여주었다고 한다. 저 발전된 구조로 성능을 높일 수 있을것이라 기대하며 experiments에서 각 tag를 따로 사용하였는데 multiple tag를 동시에 사용할 경우 좋은 결과가 나올 것이라고 암시한다.
