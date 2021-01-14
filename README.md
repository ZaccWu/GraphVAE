# GraphVAE

## Loss function
**AutoEncoder**:
$$\mathcal{L}_{AE}=norm\times[-\log(sigmoid(x))wz-\log(1-sigmoid(x))(1-z)]$$

其中$w$是weight，$z$是target，$x$是prediction。

**Variational AutoEncoder**（对于D维隐变量）:
$$\mathcal{L}_{VAE}=\mathcal{L}_{AE}-\frac{1}{2\times N_{node}}\times[\sum_D(1+2\log\sigma-\mu^2-\sigma^2)]$$

