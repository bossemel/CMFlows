set -e
echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_uniform \
--epochs 100 \
--batch-size 128 \
--marginal uniform \
--obs 10000 \
--mix \
--disable_marginal \
--error_bars
