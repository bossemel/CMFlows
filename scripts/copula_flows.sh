set -e
echo 'Begin clayton Copula Experiment - Sigmoid'

python RealNVP.py \
--exp_name clayton_10000_100_sigmoid \
--epochs 200 \
--copula clayton \
--theta 2 \
--batch-size 3000 \
--test-batch-size 3000 \
--obs 10000 \
--random_seed 58093 \
--grid_search \
--early_stopping

stop

echo 'Begin frank Copula Experiment - Sigmoid'

python RealNVP.py \
--exp_name frank_10000_100_sigmoid \
--epochs 200 \
--copula frank \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 10000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct sigmoid

echo 'Begin gumbel Copula Experiment - Sigmoid'

python RealNVP.py \
--exp_name gumbel_10000_100_sigmoid \
--epochs 200 \
--copula gumbel \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 10000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct sigmoid

echo 'Begin clayton Copula Experiment - Gaussian'

python RealNVP.py \
--exp_name clayton_10000_100_gaussian \
--epochs 200 \
--copula clayton \
--theta 2 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 10000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct gaussian

echo 'Begin frank Copula Experiment - gaussian'

python RealNVP.py \
--exp_name frank_10000_100_gaussian \
--epochs 200 \
--copula frank \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 10000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct gaussian

echo 'Begin gumbel Copula Experiment - gaussian'

python RealNVP.py \
--exp_name gumbel_10000_100_gaussian \
--epochs 200 \
--copula gumbel \
--theta 5 \
--batch-size 3000 \
--test-batch-size 3000 \
--num-blocks 5 \
--obs 10000 \
--num_hidden 64 \
--random_seed 58093 \
--transform_fct gaussian
