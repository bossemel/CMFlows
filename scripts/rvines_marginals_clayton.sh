set -e

echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_clayton_gaussian_marginal_nsf \
--epochs 100 \
--batch-size 100 \
--marginal gaussian \
--copula clayton \
--mu 0 \
--var 1 \
--obs 10000


echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_clayton_gamma_nsf \
--epochs 100 \
--batch-size 100 \
--marginal gamma \
--mu 0 \
--var 1 \
--obs 10000 \
--copula clayton
