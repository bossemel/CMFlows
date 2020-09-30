set -e
echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_uniform_error_4_256 \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal uniform \
--obs 10000 \
--error_bars \
--random_seed 32 \
--num_hidden_RealNVP 256 \
--num-blocks 4

# echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

# python RVine.py \
# --exp_name RVine_gumbel_gamma_error_new \
# --epochs 100 \
# --batch-size 100 \
# --copula gumbel \
# --marginal gamma \
# --obs 10000 \
# --error_bars
