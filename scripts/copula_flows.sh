set -e
echo 'Begin clayton Copula Experiment - Sigmoid'

python RealNVP.py \
--exp_name clayton_500000_100_sigmoid \
--epochs 100 \
--copula clayton \
--theta 2 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 5000000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct sigmoid

echo 'Begin frank Copula Experiment - Sigmoid'

python RealNVP.py \
--exp_name frank_500000_100_sigmoid \
--epochs 100 \
--copula frank \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 500000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct sigmoid

echo 'Begin gumbel Copula Experiment - Sigmoid'

python RealNVP.py \
--exp_name gumbel_500000_100_sigmoid \
--epochs 100 \
--copula gumbel \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 500000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct sigmoid

echo 'Begin clayton Copula Experiment - Gaussian'

python RealNVP.py \
--exp_name clayton_500000_100_gaussian \
--epochs 100 \
--copula clayton \
--theta 2 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 5000000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct gaussian

echo 'Begin frank Copula Experiment - gaussian'

python RealNVP.py \
--exp_name frank_500000_100_gaussian \
--epochs 100 \
--copula frank \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 500000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct gaussian

echo 'Begin gumbel Copula Experiment - gaussian'

python RealNVP.py \
--exp_name gumbel_500000_100_gaussian \
--epochs 100 \
--copula gumbel \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 500000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct gaussian
