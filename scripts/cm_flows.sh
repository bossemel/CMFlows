set -e
# echo 'Begin Gaussian Marginal Experiment'

# python CM_Flow.py \
# --exp_name CM_Flow_Gaussian \
# --epochs 100 \
# --copula clayton \
# --marginal_1 gaussian \
# --marginal_2 gaussian \
# --batch-size 100 \
# --test-batch-size 100 \
# --random_seed 58093 \
# --theta 2 \
# --mu -2 \
# --var 3 \
# --obs 10000 \
# --num_flow_layers_DDSF 5 \
# --num_hid_layers_DDSF 2 \
# --num_ds_dim 8 \
# --num_ds_layers 1 \
# --dimh_DDSF 64

echo 'Begin CM_Flow_Clayton_Gamma'

python CM_Flow.py \
--exp_name CM_Flow_Clayton_Gamma \
--epochs 1 \
--copula clayton \
--marginal_1 gamma \
--marginal_2 gamma \
--batch-size 100 \
--test-batch-size 100 \
--random_seed 58093 \
--theta 2 \
--alpha 5 \
--obs 10000 \
--num_flow_layers_DDSF 5 \
--num_hid_layers_DDSF 2 \
--num_ds_dim 8 \
--num_ds_layers 1 \
--dimh_DDSF 64 \
--early_stopping \
--clip_grad_norm \
--pretrain_models

# echo 'Begin CM_Flow_Clayton_Gamma'

# python CM_Flow.py \
# --exp_name CM_Flow_Clayton_Gamma \
# --epochs 100 \
# --copula clayton \
# --marginal_1 lognormal \
# --marginal_2 lognormal \
# --batch-size 100 \
# --test-batch-size 100 \
# --random_seed 58093 \
# --theta 2 \
# --mu 5 \
# --var 2 \
# --obs 10000 \
# --num_flow_layers_DDSF 5 \
# --num_hid_layers_DDSF 2 \
# --num_ds_dim 8 \
# --num_ds_layers 1 \
# --dimh_DDSF 64
