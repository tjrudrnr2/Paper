### Paper : [https://arxiv.org/abs/1609.03126](https://arxiv.org/abs/1609.03126)

# Introduction

 EBGAN은 GAN에 Energy라는 개념을 도입한 모델이다. supervised learning에서 data X와 label Y의 쌍 (X,Y)가 옳다면 낮은 energy를, 틀리다면 높은 energy를 assign하여 energy surface를 구성해나간다. 이를 unsupervised learning으로 생각해보면 data manifold에 가까우면 낮은 energy를 갖는다고 할 수 있다. 따라서, EBGAN에서 Discriminator는 Generator가 생성한 이미지에 높은 energy를 assign하는 Energy function 역할을 한다.

⇒ Data manifold를 잘 정의할 수 있다

# Objective Function

 $L_D(x,z)=D(x)+[m-D(G(z))]^+ \qquad (1)$
 
 $L_G(z)=D(G(z)) \qquad (2)$

Objective function을 먼저 살펴보자. 기존 GAN처럼 $min_Gmax_D$ 방식인데, 각자 다른 loss를 가지는 것이 차이점이다. (이로 인해서 generator가 수렴을 못할 때 도움을 줄 수 있다고 함.) 

 Eq.(2)에서 Generator는 생성한 이미지에 대해서 Discriminator가 낮은 에너지를 주도록 minimizing 한다. Discriminator는 Eq.(1)의 두번째 항을 최대화함으로써 maximizing 한다. 이 때, fake image의 Energy가 margin보다 작은 경우에만 gradient를 갖는다.

# EBGAN

![image](https://user-images.githubusercontent.com/70709889/180705306-c5320d77-c45e-403a-af71-cd2b283dd2c2.png)

Discriminator 대신 Auto-Encoder를 도입한 것이 특징이다.

$D(x)=||Dec(Enc(x))-x||$

0과 1 사이의 값을 내뱉던 기존의 discriminator는 다른 samples에 대해 orthogonal하게 간주하기 때문에 유연하지 못했다. 따라서, Discriminator를 Audo-Encoder로 대체하여 위와 같은 Reconstruction loss를 갖는 것이다. 그런데 Auto-Encoder를 학습할 때, 대부분의 공간에 zero energy로 mapping 해버리는 문제가 있는데 이를 해결하기 위해서는 manifold 바깥에 high energy를 mapping 할 필요성이 있고, 이를 Generator가 fake image (contrastive samples)를 갖다바침으로써 해결할 수 있는 것이다. 따라서, Generator를 regularizer로 볼 수 있음.

# Repelling Regularizer

Mode Collapse를 방지하기 위해 제안한 regularizer이다. 따라서, Generator 학습 시에만 사용된다.

$f_{PT}(S)=\frac{1}{N(N-1)}\sum_i \sum_{j\neq i}(\frac{S_t^T S_j}{||S_i||||S_j||})^2$

S는 Encoder의 output으로, 압축된 feature를 담고 있다. N개의 batch마다 각 feature 간에 코사인 유사도를 구하는건데 N(N-1)로 왜 나눠주는거지? 아마 비슷한 feature를 참고하지 못하게 코사인 유사도를 0 (직교)에 가깝게 만들어 주기 위해서…?

# Results

EBGAN의 장점은 학습 시의 안정성과 high resolution 이미지의 생성이다.

![image](https://user-images.githubusercontent.com/70709889/180705473-973c57a8-130b-44f2-8579-ce060dbf6af9.png)

$I^{'}=E_x KL(p(y) ||p(y|x))^2$

Inception score를 살짝 수정하여 GAN과 비교해봤을 때, EBGAN이 더 다양하고 풍부한 이미지를 생성해냈음을 알 수 있다.

- semi-supervised learning
    1. EBGAN이 semi-supervised learning이 가능하도록 하기 위해서 margin value m을 점차 decay 해가는 방법이 있다. G와 D가 optimal 하다면 data manifold의 바깥 영역에는 contrastive samples가 존재하지 않을 것이고, m=0을 만족할 것이다. 이 가정 하에 margin을 낮춰가며 학습을 시키는 것이다.
    2. 두번째 방법은 Ladder network를 적용하는 것이다. contrastive samples의 hierarchical한 정보를 전달함으로써 regularizer 역할을 한다.

![image](https://user-images.githubusercontent.com/70709889/180705515-81b524db-f50e-429b-bf35-2cd2fe9284b9.png)

EBGAN 안에 LN 구조를 적용했을 때 기존 LN보다 더 낮은 error rate를 보인다.

그 외 결과들

![image](https://user-images.githubusercontent.com/70709889/180705636-a4a77e20-e0a9-4df1-bbbc-5a0214e53d10.png)

(좌) : DCGAN, (우) : EBGAN-PT

![image](https://user-images.githubusercontent.com/70709889/180705720-61afcd9f-adf2-40f6-aea5-40fbf9b8e490.png)

# Conclusion

GAN과 Auto-Encoder를 결합하여 energy-based framework를 구상했으며, 학습의 안정성 과 고해상도 면에서 좋은 성능을 보인다.

# Theorem & Lemma

- Lemma 1.
    - $a, b \geq0\; \varphi(y)=ay+b[m-y]^+$일 때, $[0, +\infty)$에서 $\varphi$의 최소값이 존재하고 그 값은 a<b에 m일 것이고 아닐 경우에는 0일 것이다.
    
    위 식을 미분하면 [0,m)에서 $\varphi^{'}(y)=a-b$이고 $(m,+\infty)$에서 $\varphi^{'}(y)=a$이다.
    
    1) a < b
    위의 기울기에 의해 $[0,m)$에서는 감소, $(m,+\infty)$에서는 증가할 것이다. 따라서, 최소값은 m이다.
    
    2) a≥ b
    
    모든 구간에서 증가 함수이므로 최소값은 0이다.
    
- Lemma 2.
    - p, q가 probability density일 때, $\int_x l_{p(x)<q(x)}dx=0$이라면 $\int_x l_{p(x) \neq(x)}dx=0$ 이다.
    

$V(G,D)=\int_{x,z}L_D(x,z)p_{data}(x)p_z(z)dxdz \qquad (3)$

$U(G,D)=\int_z L_G(z)p_z(z)dz \qquad (4)$

V : Discriminator, U : Generator

- Theorem 1
    - 내쉬평형 상태의 $(D^*, G^*)$일 때, 모든 영역에서 $p_{G^*}=p_{data}$이고 $V(D^*, G^*)=m$이다.
    
    V에 D의 목적함수를 대입해보면 
    $V(G^*, D)=\int_x (p_data(x)D(x)+p_{G^*}(x)[m-D(x)]^+)dx \qquad (6)$
    
    와 같이 나온다.
    
- Theorem 2
    - 내쉬평형 상태에서 (a) $p_{G^*}=p_{data}$, (b) $D^*(x)=\gamma$를 만족하는 $\gamma \in [0,m]$ 이 존재한다
- 수식
    
    [Energy-based Generative Adversial Networks.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3701fcdd-0265-48d9-8f66-1d815563a0df/Energy-based_Generative_Adversial_Networks.pdf)
    

# To discuss

# References

- 내시평형
    - [http://www.aistudy.co.kr/economics/nash_equilibrium.htm](http://www.aistudy.co.kr/economics/nash_equilibrium.htm)
    - 게임의 참가자들이 서로의 전략을 알고 최적의 방법을 선택하기를 반복하면 어떤 평형점에 도달하게 된다는 이론. 본 논문에서는 GAN을 게임 이론에 빗대어 설명한다.
- Margin Loss (Rangin Loss)
    - [https://hyeonnii.tistory.com/277](https://hyeonnii.tistory.com/277)
    - input 간의 상대적인 거리를 예측하기 위한 loss. metric learning이라고도 불림.
- Ladder Network (LN)
    - [https://koreapy.tistory.com/1222](https://koreapy.tistory.com/1222)
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c4713f52-50c6-4ca8-9ab1-3d239c544391/Untitled.png)
    
    - semi-superviesd learning에서 hierarchical latent variable을 반영하기 위한 모델. 인코더와 디코더를 평행하게 구성.
- Data manifold
    - [https://velog.io/@xuio/TIL-Data-Manifold-학습이란](https://velog.io/@xuio/TIL-Data-Manifold-%ED%95%99%EC%8A%B5%EC%9D%B4%EB%9E%80)
    - 고차원의 데이터 분포를 잘 이해하기 위한 저차원의 subspace로 압축한다
- Cosine Similarity
    - [https://bkshin.tistory.com/entry/NLP-8-문서-유사도-측정-코사인-유사도](https://bkshin.tistory.com/entry/NLP-8-%EB%AC%B8%EC%84%9C-%EC%9C%A0%EC%82%AC%EB%8F%84-%EC%B8%A1%EC%A0%95-%EC%BD%94%EC%82%AC%EC%9D%B8-%EC%9C%A0%EC%82%AC%EB%8F%84)
    - 벡터 간의 사잇각으로 유사도 측정
