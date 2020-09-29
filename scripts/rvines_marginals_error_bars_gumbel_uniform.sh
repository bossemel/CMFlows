set -e
echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_uniform_error_new \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal uniform \
--obs 10000 \
--error_bars \
--continue_error_bars 1

# echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

# python RVine.py \
# --exp_name RVine_gumbel_gamma_error_new \
# --epochs 100 \
# --batch-size 100 \
# --copula gumbel \
# --marginal gamma \
# --obs 10000 \
# --error_bars
