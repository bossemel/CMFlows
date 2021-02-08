set -e
echo 'Begin Copula: Clayton Marginal: uniform Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_clayton_mix_lognormal_ddsfhp \
--epochs 100 \
--batch-size 128 \
--copula clayton \
--marginal_1 mix_lognormal \
--marginal_2 mix_lognormal \
--alpha 5 \
--low 0 \
--high 1 \
--theta 2 \
--obs 10000 \
--conditional \
--marg_flow DDSF \
--cop_flow NSF \
--tail_bound_c 64 \
--num_flow_layers_DDSF 2 \
--num_hid_layers_DDSF 3 \
--dimh_DDSF 1 \
--num_ds_dim 8 \
--num_ds_layers 1 \
--weight_decay_m 1e-05 \
--lr_m 0.0001
