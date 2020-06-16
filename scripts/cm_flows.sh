set -e
echo 'Begin Gaussian Marginal Experiment'

python CM_Flow.py \
--exp_name CM_Flow_Gaussian \
--epochs 100 \
--copula CLAYTON \
--marginal GAUSSIAN \
--batch-size 1000 \
--test-batch-size 1000 \
--obs 100000 \
--random_seed 58093 \
--theta 2

echo 'Begin Uniform Copula Experiment'

python CM_Flow.py \
--exp_name CM_Flow_Uniform \
--epochs 100 \
--copula FRANK \
--marginal GAUSSIAN \
--batch-size 1000 \
--test-batch-size 1000 \
--obs 100000 \
--random_seed 58093 \
--theta 2


echo 'Begin Gamma Copula Experiment'

python CM_Flow.py \
--exp_name CM_Flow_Gamma \
--epochs 100 \
--copula GUMBEL \
--marginal GAUSSIAN \
--batch-size 1000 \
--test-batch-size 1000 \
--obs 100000 \
--random_seed 58093 \
--theta 2

