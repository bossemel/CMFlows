set -e


echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_uniform_4_256_nom \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal uniform \
--obs 10000 \
--disable_marginal \
--error_bars \
--random_seed 32 \
--num_hidden_RealNVP 256 \
--num-blocks 4
