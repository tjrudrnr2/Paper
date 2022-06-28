# Attention Is All You Need
### paper : https://arxiv.org/abs/1706.03762

## 1. Abstract & Introduction
기존 SOTA 모델들은 recurrence와 convolution으로 인해 parallelizable이 어렵고, 긴 sequence에 취약하며, memory 및 학습 시간 비용이 크다는 단점이 있었다.

recurrence와 convolution을완전히 제외하고 attention에만 기반한 Transformer를 제안했고 위의 단점들을 해결함과 동시에 SOTA를 기록했다. 또한, 구문 분석등 다양한
task에서도 성능이 높았다.

## 2. Background
sequential 연산량의 증가는 멀리 떨어진 단어들 간의 학습을 어렵게 했기 때문에 Transformer는 연산량을 상수번으로 줄였다. 이 때, effective resolution이 감소하는데 이를 Multi-Head Attention으로 해결하였다.

## 3. Model Architecture
![image](https://user-images.githubusercontent.com/70709889/176033237-bcc03398-0f12-42d5-9173-d233f9ef52be.png)

- Encoder
  - N개의 layer가 stack된 형태
  - 각 layer는 Multi-Head Attention과 Feed Forward로 이루어짐
  - layer normalization 이후 residual connection
  - residual connection을 위해 model의 모든 sub-layer을 512로 유지
- Decoder
  - N개의 layer가 stack된 형태
  - encoder와 같은 2개의 sub-layer + encoder의 output을 받는 Multi-Head Attention
  - 마찬가지로 residual connection
  - self-attention에서 position i에서 이후의 position을 참조하는 것을 방지하기 위해 Masked 적용
- Attention

3가지 방식의 attention이 있다.
1. encoder-decoder attention
decoder로부터 query를 받고 encoder의 output으로부터 key와 value를 받는다. 이것은 decoder가 input sequence의 모든 position을 참고하도록 해준다.
2. self-attention in encoder
말 그대로 자기 자신을 attention한다. key, value, query가 encoder의 이전 layer로부터 오며 모두 동일하다.
3. self-attention in decoder (Masked)
decoder의 position 정보가 leftward로 흐르면 안되기 때문에 이를 방지하기 위해 $-\infty$ 로 masking 해준다. 그러면 softmax의 input으로 $-\infty$가 들어가기 때문에 미래 시점의 단어들이 0에 수렴하게 된다.

  - Scaled Dot-Product Attention
![image](https://user-images.githubusercontent.com/70709889/176039169-9bdfa9c2-b4d7-45ef-a7f0-ea009fd2f548.png)

    - $Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$
    - $d_{k}$로 나눠주는 이유는 products 값이 너무 커지게 될 경우 softmax에서 gradient가 너무 작은 값을 가지기 때문
    
![image](https://user-images.githubusercontent.com/70709889/176039860-cf8b4c2c-f462-4431-8ac3-028a3b10d136.png)
![image](https://user-images.githubusercontent.com/70709889/176040649-18e5461a-9dbc-4a22-b438-cec23b2b4daa.png)

  - Multi-HEAD Attention
    - single attention을 사용하는 것보다 h개의 다른 linear projection을 통과시켜 각각 attention을 수행한 뒤, concat하여 projected할 때 더 성능이 좋았다.
    - 다른 position의 representation subspace 정보를 joint하는 역할
    - 앞서 embedding의 size를 512로 하였고 본 논문에서는 8개의 head로 진행하였기 때문에 각 head마다 key, value, model의 size는 64가 된다
- Position-wise Feed-Forward Networks
  - $FFN(x)=max(0,xW_{1}+b_{1})W_{2}+b_{2}$
  - 두번의 선형변환과 ReLU함수로 이루어져 있다.
  - 각 layer는 다른 파라미터를 사용한다. kernel size가 1인 두개의 convolution을 생각해볼 수 있으며 input과 output은 512 dimension이고 inner-layer는 2048의 차원을 가진다.
- Embeddings and Softmax
  - input token과 output token을 512-dim의 벡터로 convert하기 위해 학습가능한 embedding을 사용
  - decoder output에서 next-token을 예측하는 확률로 바꾸기 위해 학습가능한 linear transformation과 softmax 사용
  - 두 embedding layer와 softmax 이전의 linear transformation 사이에 가중치 행렬 공유
  - embedding layer에서 가중치들에 $\sqrt{d_{model}}$을 곱해주었다
  
  ![image](https://user-images.githubusercontent.com/70709889/176045581-a232e2f2-bdc5-438e-b7bd-4d715d68111e.png)

- Positional Encoding
  - recurrence와 convolution을 제거했기 때문에 sequence의 순서를 알려줄 필요가 있다
  - encoder와 decoder의 input embedding에 **positinal encoding** 추가
  - input과 sum되어야 하므로 당연히 크기는 $d_{model}$ dimension으로 같다
  - 다른 positional embedding을 사용해도 되지만 긴 sequence에 유리한 sinusoid를 사용하였다
## 4. Why Self-Attention
- layer당 연산량 감소
- 병렬화 가능
- long-range dependency의 sequence도 학습가능
- model의 해석가능성
- single attention은 다양한 task에서도 잘 작동했고 multi-head는 sentence의 구문이나 sementic한 구조도 학습가능

![image](https://user-images.githubusercontent.com/70709889/176091465-94743c65-7a53-490a-90c9-4a3f59f62ecd.png)

위의 그림에서 알 수 있듯이 Self-Attention이 recurrent와 convolution과 비교했을 때 complexity와 sequential operation이 뛰어나다. 표의 오른쪽 column을 보면 maximum path도 짧은 것을 확인할 수 있는데 이는 forward와 backward 간에 거리가 가깝다는 것을 의미하고 path가 짧다면 long dependency를 학습하기 용이하다. (?)
## 5. Training
- 데이터 셋, Hardward, Adam 얘기는 생략하겠음
- Regularization
  - Residual Dropout
  - Label Smoothing
## 6. Results
![image](https://user-images.githubusercontent.com/70709889/176095028-27f21d89-328e-4df8-a146-771999b66e68.png)

다양한 task에서 SOTA를 달성하였고 Training Cost 또한 대폭 감소하였다.
## 7. Conclusion
attention만을 고려한 최초의 모델인 transformer를 제시하였고 encoder-decoder 구조를 유지하면서 multi-head attention, self-attention을 도입하였다.
English-German, English-French등 다양한 task에서 높은 성능을 보였고   image, audio, video등의 task에도 적용할 수 있을 것으로 예상한다.

![image](https://user-images.githubusercontent.com/70709889/176092935-623425c3-ab51-444c-a829-9e28a8328377.png)

위의 그림은 attention을 시각화한 모습이다.

## References
- [What is Attention mechanism?](https://wikidocs.net/22893)
  - 각 단어가 어떤 단어에 비중을 두는지 계산
- [Beam Search](https://velog.io/@nawnoes/%EC%9E%90%EC%97%B0%EC%96%B4%EC%B2%98%EB%A6%AC-Beam-Search)
  - 모델이 출력한 확률 분포 중에서 확률이 가장 높은 sequence를 찾기 위한 방법이다. 다양한 방법이 있지만 그 중 Beam Search는 다음 step 중에서 하이퍼파라미터 k개로 탐색 범위를 유지하는 방법이다. 빔이 클수록 target을 맞출 확률이 높아지지만 디코딩 속도가 느려진다고 한다.
  - 본 논문에서는 beam size를 4로 하였다.
- [Label Smmoothing을 하는 이유?](https://3months.tistory.com/465)
  - class score를 0 또는 1처럼 예측하는 것이 아니라 0.2, 0.9처럼 smooth하게 만들어 예측에 대한 확신을 줄인다. 
  - Why? 
    - mislabeling에 대한 고려
    - regularization, calibration에 도움
- [Calibration 이란?](https://3months.tistory.com/490)
  - confidence를 예측에 반영하는 것. ex) class "dog"이 0.8라면 80% 확률로 dog라고 확신한다!
