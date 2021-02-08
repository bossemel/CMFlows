set -e
echo 'Begin Conditional Clayton Copula Experiment'

python RealNVP.py \
--exp_name clayton_conditional  \
--epochs 100 \
--batch-size 128 \
--copula clayton \
--theta 2 \
--obs 10000 \
--conditional_copula \
--error_bars \
--num_hidden_RealNVP 256 \
--num-blocks 4

echo 'Begin Conditional Frank Copula Experiment'

python RealNVP.py \
--exp_name frank_conditional  \
--epochs 100 \
--batch-size 128 \
--copula frank \
--theta 5 \
--obs 10000 \
--conditional_copula \
--error_bars \
--num_hidden_RealNVP 256 \
--num-blocks 4

echo 'Begin Conditional Gumbel Copula Experiment'

python RealNVP.py \
--exp_name gumbel_conditional \
--epochs 100 \
--batch-size 128 \
--copula gumbel \
--theta 5 \
--obs 10000 \
--conditional_copula \
--error_bars \
--num_hidden_RealNVP 256 \
--num-blocks 4
