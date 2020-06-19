set -e
echo 'Begin gaussian Marginal Experiment - Evaluate ds dim'

python DDSF.py \
--exp_name DDSF_gaussian_ds_dim_8 \
--epochs 50 \
--marginal gaussian \
--mu -2 \
--var 3 \
--batch-size 100 \
--test-batch-size 100 \
--obs 10000 \
--random_seed 58093 \
--num_flow_layers_DDSF 5 \
--num_hid_layers_DDSF 2 \
--num_ds_dim 8 \
--num_ds_layers 1 \
--dimh_DDSF 64


#echo 'Begin Uniform Copula Experiment'

python DDSF.py \
--exp_name DDSF_gaussian_ds_dim_16 \
--epochs 50 \
--marginal gaussian \
--mu 2 \
--var 3 \
--batch-size 100 \
--test-batch-size 100 \
--obs 10000 \
--random_seed 58093 \
--num_flow_layers_DDSF 5 \
--num_hid_layers_DDSF 2 \
--num_ds_dim 16 \
--num_ds_layers 1 \
--dimh_DDSF 64

#echo 'Begin Gamma Copula Experiment'

python DDSF.py \
--exp_name DDSF_gaussian_ds_dim_32 \
--epochs 50 \
--marginal gaussian \
--mu 2 \
--var 3 \
--batch-size 100 \
--test-batch-size 100 \
--obs 10000 \
--random_seed 58093 \
--num_flow_layers_DDSF 5 \
--num_hid_layers_DDSF 2 \
--num_ds_dim 32 \
--num_ds_layers 1 \
--dimh_DDSF 64

echo 'Begin gaussian Marginal Experiment - Evaluate ds layers'

python DDSF.py \
--exp_name DDSF_gaussian_ds_layers_8 \
--epochs 50 \
--marginal gaussian \
--mu 2 \
--var 3 \
--batch-size 1000 \
--test-batch-size 1000 \
--obs 1000000 \
--random_seed 58093 \
--num_flow_layers_DDSF 5 \
--num_hid_layers_DDSF 2 \
--num_ds_dim 8 \
--num_ds_layers 1 \
--dimh_DDSF 64


#echo 'Begin Uniform Copula Experiment'

python DDSF.py \
--exp_name DDSF_gaussian_ds_layers_16 \
--epochs 50 \
--marginal gaussian \
--mu 2 \
--var 3 \
--batch-size 1000 \
--test-batch-size 1000 \
--obs 1000000 \
--random_seed 58093 \
--num_flow_layers_DDSF 5 \
--num_hid_layers_DDSF 2 \
--num_ds_dim 16 \
--num_ds_layers 1 \
--dimh_DDSF 64

#echo 'Begin Gamma Copula Experiment'

python DDSF.py \
--exp_name DDSF_gaussian_ds_layers_32 \
--epochs 50 \
--marginal gaussian \
--mu 2 \
--var 3 \
--batch-size 1000 \
--test-batch-size 1000 \
--obs 1000000 \
--random_seed 58093 \
--num_flow_layers_DDSF 5 \
--num_hid_layers_DDSF 2 \
--num_ds_dim 32 \
--num_ds_layers 1 \
--dimh_DDSF 64
