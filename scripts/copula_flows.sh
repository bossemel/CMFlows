set -e
echo 'Begin Clayton Copula with best Hyperparameters'

python NSF.py \
--exp_name clayton  \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--theta 2 \
--obs 10000 \
--conditional_copula #\
#--error_bars

echo 'Begin Frank Copula with best Hyperparameters'

python NSF.py \
--exp_name frank  \
--epochs 100 \
--batch-size 100 \
--copula frank \
--theta 5 \
--obs 10000 \
--conditional_copula
#\
#--error_bars

echo 'Begin Gumbel Copula with best Hyperparameters'

python NSF.py \
--exp_name gumbel \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--theta 5 \
--obs 10000 \
--conditional_copula \

#--error_bars
