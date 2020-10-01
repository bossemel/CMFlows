set -e


# echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

# python RVine.py \
# --exp_name RVine_gumbel_gaussian_error_new \
# --epochs 100 \
# --batch-size 100 \
# --copula gumbel \
# --marginal gaussian \
# --obs 10000 \
# --error_bars

echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_gamma_error_new_4_256 \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal gamma \
--obs 10000 \
--error_bars \
--num_hidden_RealNVP 256 \
--num-blocks 4
