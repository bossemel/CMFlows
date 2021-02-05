set -e
echo 'Begin Copula: clayton Marginal: uniform Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_clayton_mix_gamma_ddsfhp \
--epochs 100 \
--batch-size 128 \
--copula clayton \
--marginal_1 mix_gamma \
--marginal_2 mix_gamma \
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
--num_hid_layers_DDSF 4 \
--dimh_DDSF 16 \
--num_ds_dim 4 \
--num_ds_layers 3 \
--weight_decay_m 0.0001 \
--lr_m 0.00001
