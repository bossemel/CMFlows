set -e
echo 'Begin Conditional Clayton Copula Experiment'

python RealNVP.py \
--exp_name clayton_conditional  \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--theta 2 \
--obs 10000 \
--conditional_copula \
--error_bars

echo 'Begin Conditional Frank Copula Experiment'

python RealNVP.py \
--exp_name frank_conditional  \
--epochs 100 \
--batch-size 100 \
--copula frank \
--theta 5 \
--obs 10000 \
--conditional_copula \
--error_bars

echo 'Begin Conditional Gumbel Copula Experiment'

python RealNVP.py \
--exp_name gumbel_conditional \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--theta 5 \
--obs 10000 \
--conditional_copula \
--error_bars
