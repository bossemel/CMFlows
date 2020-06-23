set -e
echo 'Begin Copula: Clayton Marginal: Gamma Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_Clayton_Gamma_pretr_only \
--epochs 100 \
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

echo 'Begin Copula: Clayton, Margina: Gamma, CM Flow only'

python CM_Flow.py \
--exp_name CM_Flow_Clayton_Gamma_cm_flow_only \
--epochs 100 \
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
--train_cm_flow

echo 'Begin Copula: Clayton, Margina: Gamma, Pretrain and CM Flow'

python CM_Flow.py \
--exp_name CM_Flow_Clayton_Gamma_pretr_cm \
--epochs 100 \
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
--train_cm_flow \
--pretrain_models
