set -e
echo 'Begin Clayton Copula Experiment - Sigmoid True'

python RealNVP.py \
--exp_name Clayton_500000_100 \
--epochs 100 \
--dataset CLAYTON \
--theta 2 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 500000 \
--num_hidden 64 \
--random_seed 58093

echo 'Begin Frank Copula Experiment - Sigmoid True'

python RealNVP.py \
--exp_name Frank_500000_100 \
--epochs 100 \
--dataset FRANK \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 500000 \
--num_hidden 64 \
--random_seed 58093

echo 'Begin Gumbel Copula Experiment - Sigmoid True'

python RealNVP.py \
--exp_name Gumbel_500000_100 \
--epochs 100 \
--dataset GUMBEL \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 500000 \
--num_hidden 64 \
--random_seed 58093
