Noise-Contrastive Estimation NCE

NCE는 진짜 데이터와 인공적인 노이즈 데이터를 구별하는 문제를 풀도록 모델을 훈련시켜 생성 모델을 학습하는 기법임.
NCE는 GAN처럼 진짜와 가짜를 구별하는 이진 분류문제를 푸는점에서 유사하지만, NCE는 별도의 Discriminator가 없고, 대신 사전에 정의된 노이즈 분포를 사용하여 모델을 훈련시킨다는 점에서 차이가 있음.