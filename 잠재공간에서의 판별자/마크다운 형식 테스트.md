$$ D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$

$$ \nabla_x \log \frac{D^*(x)}{1 - D^*(x)} = \underbrace{\nabla_x \log p_{data}(x)}_{\text{Data Score}} - \underbrace{\nabla_x \log p_g(x)}_{\text{Gen Score}} $$

$\sqrt{\eta}\xi_t$