set -e
echo 'Begin Clayton Copula Grid Search'

python RealNVP.py \
--exp_name clayton_grid_search \
--epochs 50 \
--copula clayton \
--theta 2 \
--batch-size 100 \
--obs 10000 \
--grid_search

echo 'Begin Frank Copula Grid Search'

python RealNVP.py \
--exp_name frank_grid_search \
--epochs 50 \
--copula frank \
--theta 5 \
--batch-size 100 \
--obs 10000 \
--grid_search

echo 'Begin Gumbel Copula Grid Search'

python RealNVP.py \
--exp_name gumbel_grid_search \
--epochs 500 \
--copula gumbel \
--theta 5 \
--batch-size 100 \
--obs 10000 \
--grid_search
