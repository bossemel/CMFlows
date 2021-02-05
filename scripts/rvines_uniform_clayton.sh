set -e


echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_uniform_nsf \
--epochs 100 \
--batch-size 128 \
--copula clayton \
--marginal uniform \
--obs 10000 \
--disable_marginal \
--random_seed 32
