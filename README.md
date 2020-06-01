# Copula and Marginal Generative Flows

## Installation

Create a local environment and install the requirements:
```
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## To run RealNVP:

To train the [RealNVP](https://arxiv.org/abs/1605.08803) run:
```
python3 RealNVP.py --dataset CLAYTON --epochs 100
```

Available datasets are copula samples from the Clayton, Frank, Gumbel, Gaussian and t-copulas.


## To run DDSF.py:

To train the [Deep Dense Sigmoidal Flow (DDSF)](https://arxiv.org/pdf/1804.00779.pdf) run:

```
python3 DDSF.py
```

Available datasets are @TODO: add datasets

### References:

RealNVP is based on https://github.com/ikostrikov/pytorch-flows. DDSF is based on https://github.com/CW-Huang/NAF/.
