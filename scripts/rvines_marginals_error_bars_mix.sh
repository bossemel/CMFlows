set -e
echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_gaussian_error \
--epochs 100 \
--batch-size 100 \
--marginal gamma \
--mu 0 \
--var 1 \
--obs 10000 \
--mix \
--error_bars

echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_gamma_error \
--epochs 100 \
--batch-size 100 \
--marginal gamma \
--mu 0 \
--var 1 \
--obs 10000 \
--mix \
--error_bars
