# CM Flows - Copula Density Estimation With Normalizing Flows

## Installation

Create a local environment and install the requirements:
```
python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt
pip3 install -e .
```

On some systems, the package ```pyvinecopulib``` has to be installed using:

```
pip3 install --user --no-binary 'pyvinecopulib' pyvinecopulib
```

## To run the copula flow:

To train the [RealNVP](https://arxiv.org/abs/1605.08803) run:
```
python3 RealNVP.py
```

Available datasets are copula samples from the Clayton, Frank or Gumbel copula. Available option can be
found in ```RealNVP_modules/options```.


## To run marginal flow:

To train the [Deep Dense Sigmoidal Flow (DDSF)](https://arxiv.org/pdf/1804.00779.pdf) run:

```
python3 DDSF.py
```

Available datasets are samples from Gaussian, Uniform, Gamma, Lognormal and bimodal Gaussian distribution.
Available option can be found in ```DDSF_modules/options```.

## To run the bivariate CM Flow:

To train the [CM Flow](https://arxiv.org/abs/1907.03361) run:

```
python3 CM_Flow.py --train_cm_flow
```

Available datasets are joint samples with Clayton, Frank or Gumbel copula and Gaussian, Uniform, Gamma and
Lognormal marginals. Available option can be found in ```CM_modules/options```.

## To run the multivariate CM Flow:

To train the multivariate CM Flow using R-Vines run:

```
python3 RVine.py
```

Available datasets are joint samples with Clayton, Frank or Gumbel copula and Gaussian, Uniform, Gamma,
Lognormal and bimodal Gaussian marginals. Available option can be found in ```RVine_modules/options```.


### References:

RealNVP is based on https://github.com/ikostrikov/pytorch-flows. DDSF is based on https://github.com/CW-Huang/NAF/.
