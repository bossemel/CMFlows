set -e
echo 'Begin Clayton Copula with best Hyperparameters'

python RealNVP.py \
--exp_name clayton_besthyp  \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--theta 2 \
--obs 10000

echo 'Begin Frank Copula with best Hyperparameters'

python RealNVP.py \
--exp_name frank_besthyp  \
--epochs 100 \
--batch-size 100 \
--copula frank \
--theta 5 \
--obs 10000

echo 'Begin Gumbel Copula with best Hyperparameters'

python RealNVP.py \
--exp_name gumbel_besthyp \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--theta 5 \
--obs 10000
