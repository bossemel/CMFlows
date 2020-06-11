set -e
echo 'Begin Clayton Copula Experiment - Sigmoid'

python RealNVP.py \
--exp_name Clayton_500000_100_sigmoid \
--epochs 100 \
--dataset CLAYTON \
--theta 2 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 5000000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct sigmoid

echo 'Begin Frank Copula Experiment - Sigmoid'

python RealNVP.py \
--exp_name Frank_500000_100_sigmoid \
--epochs 100 \
--dataset FRANK \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 500000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct sigmoid

echo 'Begin Gumbel Copula Experiment - Sigmoid'

python RealNVP.py \
--exp_name Gumbel_500000_100_sigmoid \
--epochs 100 \
--dataset GUMBEL \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 500000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct sigmoid

echo 'Begin Clayton Copula Experiment - Gaussian'

python RealNVP.py \
--exp_name Clayton_500000_100_gaussian \
--epochs 100 \
--dataset CLAYTON \
--theta 2 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 5000000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct gaussian

echo 'Begin Frank Copula Experiment - gaussian'

python RealNVP.py \
--exp_name Frank_500000_100_gaussian \
--epochs 100 \
--dataset FRANK \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 500000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct gaussian

echo 'Begin Gumbel Copula Experiment - gaussian'

python RealNVP.py \
--exp_name Gumbel_500000_100_gaussian \
--epochs 100 \
--dataset GUMBEL \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 500000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct gaussian
