set -e
echo 'Begin Gaussian Marginal Experiment'

python CM_Flow.py \
--exp_name CM_Flow_Gaussian \
--epochs 50 \
--copula clayton \
--marginal_1 gaussian \
--marginal_2 gaussian \
--batch-size 100 \
--test-batch-size 100 \
--random_seed 58093 \
--theta 2 \
--mu -2 \
--var 3 \
--obs 10000 \
--num_flow_layers_DDSF 5 \
--num_hid_layers_DDSF 2 \
--num_ds_dim 8 \
--num_ds_layers 1 \
--dimh_DDSF 64

echo 'Begin Uniform Copula Experiment'

python CM_Flow.py \
--exp_name CM_Flow_Uniform \
--epochs 50 \
--copula frank \
--marginal_1 gaussian \
--marginal_2 gaussian \
--batch-size 100 \
--test-batch-size 100 \
--obs 10000 \
--random_seed 58093 \
--theta 2


echo 'Begin Gamma Copula Experiment'

python CM_Flow.py \
--exp_name CM_Flow_Gamma \
--epochs 50 \
--copula gumbel \
--marginal_1 gaussian \
--marginal_2 gaussian \
--batch-size 100 \
--test-batch-size 100 \
--obs 10000 \
--random_seed 58093 \
--theta 2

