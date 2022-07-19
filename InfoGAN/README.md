# InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets

### Paper : [https://arxiv.org/abs/1606.03657](https://arxiv.org/abs/1606.03657)

## Summary

## Introduction

기존의 representation learning은 대부분 supervised이고  unsupervised 방식은 discrete latent만 disentangle 시킬 수 있었다. 또한, 기존 GAN은 random noise z를 아무런 restrict 없이 사용하기 때문에 disentangle이 어렵다. 따라서, 이 논문에서는 discrete, continuos latent code 모두 disentangled representation를 unsupervised로 학습하는 것을 목표로 하고 있다.

이를 위해 본 논문에서는 정보 이론을 적용하였는데, generator에 추가로 주입한 latent variables c와 G(z,c) 사이의 mutual information을 최대화하는 방향으로 학습한다.

## Mutual Information

 GAN의 entangle 문제를 해결하기 위해, 본 논문에서는 input을 두 가지로 나누었다.

(1) z : 더이상 압축할 수 없는 noise

(2) c : semantic feature를 target 할 latent code

 그러나 latent code c를 추가로 주입하려고 하는데 단순히 주입하기만 하면 c가 generation 과정에서 무시 될 수 있기 때문에 G(z,c)와 높은 mutual information을 유지해야 한다. 

![image](https://user-images.githubusercontent.com/70709889/179837312-2c42bed4-3438-4d60-b9aa-6a53776b1a17.png)

이를 본 논문에서는 generation process에서 lost된다고 표현하고 있고 generator가 생성하는 분포 $P_{G}(c|x)$가 small entropy를 가지기를 원한다고 한다. 

## Variational Mutual Information Maximization

기존 GAN의 objective fucntion은

![https://user-images.githubusercontent.com/70709889/179458060-6337eae2-e7c9-424f-af3c-707963e6b3b3.png](https://user-images.githubusercontent.com/70709889/179458060-6337eae2-e7c9-424f-af3c-707963e6b3b3.png)

인데, 여기서 mutual information I(X;Y)를 추가하여 InfoGAN의 objective function은 다음과 같다.

![https://user-images.githubusercontent.com/70709889/179456687-3cca9a05-9852-4a87-b579-d67374fba484.png](https://user-images.githubusercontent.com/70709889/179456687-3cca9a05-9852-4a87-b579-d67374fba484.png)

이 때,

 $P(c_1, c_2, ... , c_L)=\prod_{i-1}^L P(c_i)$ 

latent variables를 concat하여 c라고 표기한다. 위 식의 mutual information 부분을 전개하면

![https://user-images.githubusercontent.com/70709889/179465901-2de98833-bb2e-4f24-9f1b-5101cf4c478b.png](https://user-images.githubusercontent.com/70709889/179465901-2de98833-bb2e-4f24-9f1b-5101cf4c478b.png)

과 같이 나온다. 이 때, $P(c|x)$를 보면 x는 generate 과정의 output이고 c는 output을 만들기 위해 입력으로 주는 latent code이다. output으로 input을 구하기 어렵기 때문에 (posterior) Variational Information Maximizatio이라는 테크닉을 사용하여 P(c|x)를 Q(c|x)로 근사하여 lower bound를 구한다.

마지막 부등호에도 $P(c|x)$가 남아있기 때문에 더 작업해야함…

![image](https://user-images.githubusercontent.com/70709889/179837386-18144170-1dc0-4280-b4b7-6f30ae9823f9.png)

![image](https://user-images.githubusercontent.com/70709889/179837407-381b6766-61db-4813-9774-748187294039.png)

위 과정을 거쳐 아래와 같은 objective function을 가진다.

$min_{G,Q} max_D V_{infoGAN}(D,G,Q)=V_{GAN}(D,G)-\lambda L_1 (G,Q)$

- Proof
    
    [INFOGA_1.PDF](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1270d733-930f-4149-9b5c-fc866545e154/INFOGA_1.pdf)
    

## InfoGAN

![https://user-images.githubusercontent.com/70709889/179468274-9286546c-b836-4f95-8bfc-1f2529b5aec4.png](https://user-images.githubusercontent.com/70709889/179468274-9286546c-b836-4f95-8bfc-1f2529b5aec4.png)

학습을 쉽게 하기 위해 DCGAN을 기반으로 구현하였다. 위에서 다룬 것 처럼 $P(c|x)$를 근사한 $Q(c|x)$의 분포를 구해야한다. 이를 neural network로 parameterize 하였고 D와 layer를 공유하지만 distribution $Q(c|x)$를 뱉을 fc layer 하나만 추가하였다.

- categorical code, $c_i$
    - $Q(c_i|x)$를 나타내기 위해 softmax 사용
    - objective fucntion 의 hyperparameter인 $\lambda$를 1로 설정
- continuous code, $c_j$
    - posterior에 따라 비선형함수 결정. 보통 factored 가우시안이면 충분하다고 한다.
    - 1보다 작은 $\lambda$ 사용.

## Experiments

mutual information을 얼마나 잘 maximize 했는지, 한 factor만 조정해가며 disentangled representation을 잘 학습했는지
를 MNIST 데이터셋으로 비교해보자. 0~9까지의 숫자를 조절하는 categorical code인 $c_{1}$은 $Cat(K=10,p=0.1)$의 분포에서 sampling 했고, Rotation 과 Width를 나태내는 continuous code $c_2, c_3$은 $Unif(-1,1)$을 따랐다. 

- mutual information 비교

![https://user-images.githubusercontent.com/70709889/179456874-853ee504-d9d7-4d5b-91c0-db2b074333dd.png](https://user-images.githubusercontent.com/70709889/179456874-853ee504-d9d7-4d5b-91c0-db2b074333dd.png)

GAN에 mutual information을 maximize하지 않고 단순히 auxiliary distribution Q를 주고 InfoGAN과 lower bound를 비교한 결과, InfoGAN이 더 높았다. 따라서 InfoGAN이 mutual information을 잘 maximize 했다는 것을 알 수 있다.
이를 통해, GAN은 latent code를 generator에서 활용하지 않는다는 것을 알 수 있다.

- disentangled representation

![https://user-images.githubusercontent.com/70709889/179457866-701e9dd1-d21c-4ea2-b656-ec62ba327a64.png](https://user-images.githubusercontent.com/70709889/179457866-701e9dd1-d21c-4ea2-b656-ec62ba327a64.png)

(a) : MNIST의 0~9를 control 하기 위한 categorical code를 추가

(b) : 기존 GAN

(c) : Rotation continuous code 추가

(d) : Width continuous code 추가

기존 GAN인 (b)와 비교했을 때, discrete와 continuos code 모두 disentangle한 representation을 학습한 것을 확인할 수 있다.

또한, 학습된 $c_1$을 classifier로 사용하였을 때 95%의 정확도를 보인다고 한다. (MNIST 기준) 위의 그림의 (c)와 (d)처럼 unif distribution을 -2~2로 확장시켜도 학습이 잘 되었다고 한다.

![https://user-images.githubusercontent.com/70709889/179458678-823e91c1-3e16-425d-91be-91616f367b95.png](https://user-images.githubusercontent.com/70709889/179458678-823e91c1-3e16-425d-91be-91616f367b95.png)

![https://user-images.githubusercontent.com/70709889/179458694-a90cd5b1-aa10-4e04-809c-3fad4d0d992f.png](https://user-images.githubusercontent.com/70709889/179458694-a90cd5b1-aa10-4e04-809c-3fad4d0d992f.png)

3D image에 대해서도 잘 학습한다. $c_{1,2,3,4,5}$~$Unif(-1,1)$ 사용.

## Conclusion

결론은 GAN에 'latent code와 observation 간의 mutual information을 최대화해라.' 라는 알고리즘을 도입했더니 unlabeled 데이터로도 disentangled representation을 추출할 수 있었고 연산량은 GAN과 거의 동일했기 때문에 효율적으로 학습했다고 볼 수 있다. 또한, hyperparameter가 $\lambda$ 하나이기 때문에 학습도 쉽다고 한다. 저자들은 future work로 mutual information을 VAE에 적용, hierarchical representation 추출과 semi-supervised learning 의 성능 향상등을 제시하였다.

# To Discuss

- GAN에 latent code c를 삽입하였을 때, c가 무시되는 이유가 $P_{G}(x|c)=P_{G}(x)$를 만족하는 solution을 찾기 때문이라고 나오는데
    
    $\underset{G}{min}\underset{D}{max}V(D,G)=E_{x\sim p_{data}(x)}[log D(x|y)]+E_{z \sim p_z(z))]}[log(1-D(G(z|y)))]$
     CGAN에서 별다른 제약 term 없이 condition y를 추가하여 잘 작동하지 않나요? InfoGAN과 CGAN 모두 코드상으로 latent code c를 noise z와 concat하여 input으로 넣어주는 것으로 확인했는데, GAN에서 c가 무시된다고 언급한 이유가 궁금합니다.
    위 내용과 별개로 개인적으로 latent code c가 G(z,c) 분포와 mutual information이 낮다면 Generator가 Discriminator를 속이는 과정에서 c가 무시될 수도 있을 것 같다고는 생각했습니다…
    
- 논문에서 GAN의 문제점 설명하고 GAN의 extension 이라고 언급 + Experiments에서 GAN이랑 실컷 비교해놓고 정작 학습이 쉽다는 이유로 구현은 DCGAN으로 했다고 서술합니다. Experiments 에서도 DCGAN 기반으로 실험했는지 모르겠지만 만약 아니라면 위 논리 전개로는 신빙성이 떨어진다고 생각됩니다. 혹시 GAN과 DCGAN의 objective function이 같기 때문에 conv의 유무가 information theory 적용에 큰 영향이 없다는 전제가 깔려있는 건가요?

# References

- [정보이론](https://horizon.kias.re.kr/18474/)
    - 정보의 불확실한 정도를 엔트로피와 mutual information으로 수치화
    - 무작위 변수 X에 대한 엔트로피
    
    $H(x) = -\sum_{x}p(x)logp(x)=E[log\frac{1}{p(x)}]$
    
    - X와 Y 사이의 mutual information
        - Y가 관찰되었을 때 줄어드는 X의 불확실성 (엔트로피). 따라서 X와 Y가 독립일 경우 mutual information은 0 이다.
        또한, 교환법칙이 성립한다.
        $I(X;Y)=H(X)-H(X|Y)=H(Y)-H(Y|X)$
        - Equation
        
        $I(X;Y)=H(X)-H(X|Y)$
        
        $=-\sum_{x}p(x)logp(x)+\sum_{x,y}p(x,y)logp(x|y)$
        
        $=-\sum_{x}p(x)logp(x)+\sum_{x,y}p(x,y)log\frac{p(x,y)}{p(y)}$
        
        $=H(X)+H(Y)-H(X,Y)$
        
        - Y가 주어졌을 때 X에 대한 불확실한 정도 H(X|Y)를 빼준다
        
        ![https://user-images.githubusercontent.com/70709889/179454460-ebe4628f-5f11-4820-8f6f-d41e44bfcc1e.png](https://user-images.githubusercontent.com/70709889/179454460-ebe4628f-5f11-4820-8f6f-d41e44bfcc1e.png)
        
        ![https://user-images.githubusercontent.com/70709889/179467075-909249c0-fbdb-4679-ab0a-f5fb62c77337.png](https://user-images.githubusercontent.com/70709889/179467075-909249c0-fbdb-4679-ab0a-f5fb62c77337.png)
        
- [Monte Carlo Simulation](https://ko.wikipedia.org/wiki/%EB%AA%AC%ED%85%8C%EC%B9%B4%EB%A5%BC%EB%A1%9C_%EB%B0%A9%EB%B2%95)
    - 반복적인 random sampling을 이용하여 함수의 값을 근사하는 알고리즘
- 참고한 블로그
    - [https://blog.naver.com/egg5562/221014635751](https://blog.naver.com/egg5562/221014635751)
    - [https://jaejunyoo.blogspot.com/2017/03/infogan-1.html](https://jaejunyoo.blogspot.com/2017/03/infogan-1.html)
    - [https://github.com/sungreong/Infogan](https://github.com/sungreong/Infogan)
    - [https://en.wikipedia.org/wiki/Law_of_total_expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation)
    - [https://ko.wikipedia.org/wiki/연속균등분포](https://ko.wikipedia.org/wiki/%EC%97%B0%EC%86%8D%EA%B7%A0%EB%93%B1%EB%B6%84%ED%8F%AC)
