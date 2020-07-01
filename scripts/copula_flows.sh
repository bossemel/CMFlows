set -e
# echo 'Begin Clayton Copula with best Hyperparameters'

# python RealNVP.py \
# --exp_name clayton_besthyp  \
# --epochs 50 \
# --batch-size 1000 \
# --copula clayton \
# --theta 2 \
# --obs 50000

# echo 'Begin Frank Copula with best Hyperparameters'

# python RealNVP.py \
# --exp_name frank_besthyp  \
# --epochs 50 \
# --batch-size 1000 \
# --copula frank \
# --theta 5 \
# --obs 500000

echo 'Begin Gumbel Copula with best Hyperparameters'

python RealNVP.py \
--exp_name gumbel_besthyp  \
--epochs 1 \
--batch-size 1000 \
--copula gumbel \
--theta 5 \
--obs 500000
