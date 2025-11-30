Generative Stochastic Networks (GSN)
GSN은 Denoising AutoEncoder(DAE)를 일반화시킨 생성모델임.
작동 원리:
GSN은 샘플을 생성하거나 학습할 때, 파라미터화된 markov chain을 정의하고 이를 반복적으로 실행해야 함.
즉, RBM/DBM이 MCMC에 의존했던 것처럼 GSN도 여전히 일종의 반복적인 샘플링 과정(markov chain)을 필요로 함.

또한 GSN은 샘플링 과정에서 feedback loops를 사용하는데 이때문에 값이 폭주할 수 있는 ReLU의 이점을 활용하기 어려움.