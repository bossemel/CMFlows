set -e
echo 'Begin Clayton Copula Grid Search'

python RealNVP.py \
--exp_name clayton_grid_search \
--epochs 100 \
--copula clayton \
--theta 2 \
--batch-size 100 \
--obs 10000 \
--random_seed 58093 \
--grid_search \
--early_stopping

echo 'Begin Frank Copula Grid Search'

python RealNVP.py \
--exp_name frank_grid_search \
--epochs 100 \
--copula frank \
--theta 5 \
--batch-size 100 \
--obs 10000 \
--random_seed 58093 \
--grid_search \
--early_stopping

echo 'Begin Gumbel Copula Grid Search'

python RealNVP.py \
--exp_name gumbel_grid_search \
--epochs 1000 \
--copula gumbel \
--theta 5 \
--batch-size 100 \
--obs 10000 \
--random_seed 58093 \
--grid_search \
--early_stopping
