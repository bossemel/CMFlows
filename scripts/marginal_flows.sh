set -e
echo 'Begin Gaussian Marginal Experiment'

python DDSF.py \
--exp_name DDSF_Gaussian \
--epochs 100 \
--marginal GAUSSIAN \
--mu 2 \
--var 0.2 \
--batch-size 1000 \
--test-batch-size 1000 \
--obs 100000 \
--random_seed 58093

echo 'Begin Uniform Copula Experiment'

python DDSF.py \
--exp_name DDSF_Uniform \
--epochs 100 \
--marginal UNIFORM \
--low 2 \
--high 4 \
--batch-size 1000 \
--test-batch-size 1000 \
--obs 100000 \
--random_seed 58093

echo 'Begin Gamma Copula Experiment'

python DDSF.py \
--exp_name DDSF_Gamma \
--epochs 100 \
--marginal Gaussian \
--a_param 5 \
--batch-size 1000 \
--test-batch-size 1000 \
--obs 100000 \
--random_seed 58093
