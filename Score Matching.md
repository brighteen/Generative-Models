Score Matching
score function은 log-likelihood의 gradient로 정의됨. 

$$ \nabla_x log p(x) $$

score matching은 실제 확률분포함수를 구하는 것이 아닌, 이러한 score값을 활용하여 확률함수를 추정하는 것을 의미. 이런 스코어 매칭 기법은 최근 SDE(stochastic Differential Equation)를 확룔해 diffusion model을 모델링할 때 널리 활용됨.
p(x)를 $\theta$로 parameterize된 energy-based model로 정의할 때, score function은 다음과 같이 쓸 수 있음.

$$p_{\theta}(x) = \frac{e^{-f_{\theta}(x)}}{Z_{\theta}} ,where Z_{\theta} = \int e^{-f_{\theta}(x)}$$

이때 정규화 상수 Z가 intractable할 때가 많은데 Score function을 활용해 log우도를 계산한다고 하면 log우도에 대해 미분을 취하기 때문에 입력에 종속적인 정규화 상수가 사라져 계산이 쉬워짐.

$$ \nabla_x log p_{\theta}(x) = -\nabla_x f_{\theta}(x) $$

```
Q. 스코어함수를 로그우도에 대한 미분한다는 의미가 무엇인가?
A.

```
