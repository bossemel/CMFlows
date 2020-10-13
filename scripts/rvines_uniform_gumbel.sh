set -e


echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_uniform_nsf \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal uniform \
--obs 10000 \
--disable_marginal \
--error_bars \
--random_seed 32
