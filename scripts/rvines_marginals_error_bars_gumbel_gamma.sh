set -e

# echo 'Begin gumbel Copula Rvine: RVine_gumbel'

# python RVine.py \
# --exp_name RVine_gumbel_gaussian_error_new \
# --epochs 100 \
# --batch-size 100 \
# --copula gumbel \
# --marginal gaussian \
# --mu 0 \
# --var 1 \
# --obs 10000 \
# --error_bars

echo 'Begin gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_gamma_error_nsf \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal gamma \
--mu 0 \
--var 1 \
--obs 10000 \
--error_bars
