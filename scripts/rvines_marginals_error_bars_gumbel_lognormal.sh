set -e
echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_lognormal_error_new \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal lognormal \
--obs 10000 \
--error_bars

# echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

# python RVine.py \
# --exp_name RVine_gumbel_gamma_error_new \
# --epochs 100 \
# --batch-size 100 \
# --copula gumbel \
# --marginal gamma \
# --obs 10000 \
# --error_bars
