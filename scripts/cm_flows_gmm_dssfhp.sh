set -e
echo 'Begin Copula: Clayton Marginal: uniform Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_clayton_gmm_ddsfhp \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal_1 gmm \
--marginal_2 gmm \
--alpha 5 \
--low 0 \
--high 1 \
--theta 2 \
--obs 10000 \
--conditional \
--marg_flow DDSF \
--cop_flow NSF \
--tail_bound_c 64 \
--num_flow_layers_DDSF 10 \
--num_hid_layers_DDSF 1 \
--dimh_DDSF 128 \
--num_ds_dim 8 \
--num_ds_layers 2
