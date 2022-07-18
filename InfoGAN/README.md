# InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets

### Paper : https://arxiv.org/abs/1606.03657

## Content 
### Introduction
기존의 representation learning은 대부분 supervised였고 unsupervised 방식은 discrete latent만 disentangle
시킬 수 있었다. 또한, 기존 GAN은 random noise z를 아무런 restrict 없이 사용하기 때문에
disentangle이 어렵다. 따라서, 이 논문에서는 discrete, continuos latent code 
모두 disentangled representation를 unsupervised로 학습하는 것을 목표로 하고 있다.

이를 위해 정보 이론을 적용하였는데, generator에 추가로 주입한 latent variables c와
G(z,c) 사이의 mutual information을 최대화하는 방향으로 학습한다.

### InfoGAN
![image](https://user-images.githubusercontent.com/70709889/179468274-9286546c-b836-4f95-8bfc-1f2529b5aec4.png)

기존 GAN은 random noise z를 아무런 restriction 없이 사용되기 때문에 generator에서 entangled way로
사용될 수밖에 없다. 이를 해결하기 위해, latent code c를 추가로 주입하려고 하는데 단순히 주입하기만 하면 
c가 generation 과정에서 lost 될 수 있기 때문에 mutual information을 maximize 해야 한다. 직관적으로
생각해볼 때 c가 생성하려고 하는 이미지와 mutual information이 낮다면 generator는 이를 무시해버릴 것이다.

기존 GAN의 objective fucntion은 ![image](https://user-images.githubusercontent.com/70709889/179458060-6337eae2-e7c9-424f-af3c-707963e6b3b3.png)

인데, 여기서 mutual information I(X;Y)를 추가하여 InfoGAN의 objective function은 다음과 같다.

![image](https://user-images.githubusercontent.com/70709889/179456687-3cca9a05-9852-4a87-b579-d67374fba484.png)

이 때, latent variables $c_{i}$를 c라고 표기한다. 위 식의 mutual information 부분을 전개하면
![image](https://user-images.githubusercontent.com/70709889/179465901-2de98833-bb2e-4f24-9f1b-5101cf4c478b.png)

과 같이 나온다. 이 때, P(c|x)를 풀기 어렵기 때문에 Variational Information Maximizatio이라는 테크닉을
사용하면 P(c|x)를 Q(c|x)로 근사하여 lower bound를 구할 수 있다.

### Results
mutual information을 얼마나 잘 maximize 했는지, 한 factor만 조정해가며 disentangled representation을 잘 학습했는지
를 MNIST 데이터셋으로 비교해보자.
- mutual information 비교

![image](https://user-images.githubusercontent.com/70709889/179456874-853ee504-d9d7-4d5b-91c0-db2b074333dd.png)

InfoGAN과 GAN의 lower bound를 비교한 결과, InfoGAN이 더 높았다.
이를 통해, GAN은 latent code를 generator에서 활용하지 않는다는 것을 알 수 있다.

- disentangled representation

![image](https://user-images.githubusercontent.com/70709889/179457866-701e9dd1-d21c-4ea2-b656-ec62ba327a64.png)

(a)는 MNIST의 0~9를 control 하기 위한 categorical code를 추가하였고, (c)와 (d)는 각각 Rotation과 Width
를 학습할 continuos code이다. 기존 GAN인 (b)와 비교했을 때, discrete와 continuos code 둘 다 disentangle하여
학습한 것을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/70709889/179458678-823e91c1-3e16-425d-91be-91616f367b95.png)

![image](https://user-images.githubusercontent.com/70709889/179458694-a90cd5b1-aa10-4e04-809c-3fad4d0d992f.png)

3D image에 대해서도 잘 학습한다.

결론은 GAN에 'latent code와 observation 간의 mutual information을 최대화해라.' 라는 알고리즘을 도입했더니 unlabeled 데이터로도 disentangled
representation을 추출할 수 있었고 연산량은 GAN과 거의 동일했기 때문에 효율적으로 
학습했다고 볼 수 있다. 저자들은 future work로 
mutual information을 VAE에 적용, hierarchical representation 추출과 semi-supervised learning
의 성능 향상등을 제시하였다.


## References

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
        
        ![image](https://user-images.githubusercontent.com/70709889/179454460-ebe4628f-5f11-4820-8f6f-d41e44bfcc1e.png)
        
        ![image](https://user-images.githubusercontent.com/70709889/179467075-909249c0-fbdb-4679-ab0a-f5fb62c77337.png)

- [Monte Carlo Simulation](https://ko.wikipedia.org/wiki/%EB%AA%AC%ED%85%8C%EC%B9%B4%EB%A5%BC%EB%A1%9C_%EB%B0%A9%EB%B2%95)
    - 반복적인 random sampling을 이용하여 함수의 값을 근사하는 알고리즘
- 참고한 블로그
    - [https://blog.naver.com/egg5562/221014635751](https://blog.naver.com/egg5562/221014635751)
    - [https://jaejunyoo.blogspot.com/2017/03/infogan-1.html](https://jaejunyoo.blogspot.com/2017/03/infogan-1.html)
