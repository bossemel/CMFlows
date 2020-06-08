set -e
echo 'Begin Clayton Copula Experiment - Sigmoid True'

python RealNVP.py \
--epochs 100 \
--dataset CLAYTON \
--theta 2 \
--sigmoid \
--batch-size 100 \
--test-batch-size 1000 \
--num-blocks 5 \
--log-interval 1000 \
--obs 3000 \
--num_hidden 64 \
--random_seed 58093

echo 'Begin Frank Copula Experiment - Sigmoid True'

python RealNVP.py \
--epochs 100 \
--dataset FRANK \
--theta 5 \
--sigmoid \
--batch-size 100 \
--test-batch-size 1000 \
--num-blocks 5 \
--log-interval 1000 \
--obs 3000 \
--num_hidden 64 \
--random_seed 58093

echo 'Begin Gumbel Copula Experiment - Sigmoid True'

python RealNVP.py \
--epochs 100 \
--dataset GUMBEL \
--theta 5 \
--sigmoid \
--batch-size 100 \
--test-batch-size 1000 \
--num-blocks 5 \
--log-interval 1000 \
--obs 3000 \
--num_hidden 64 \
--random_seed 58093
