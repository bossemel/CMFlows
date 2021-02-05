set -e
echo 'Begin Copula: Clayton Marginal: uniform Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_gumbel_gmm_ddsfhp \
--epochs 100 \
--batch-size 128 \
--copula gumbel \
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
--num_flow_layers_DDSF 1 \
--num_hid_layers_DDSF 3 \
--dimh_DDSF 1 \
--num_ds_dim 8 \
--num_ds_layers 4 \
--weight_decay_m 1e-06 \
--lr_m 1e-05
