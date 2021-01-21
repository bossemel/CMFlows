set -e
echo 'Begin Copula: Clayton Marginal: uniform Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_clayton_mix_lognormal_nsfhp_uncon \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal_1 mix_lognormal \
--marginal_2 mix_lognormal \
--alpha 5 \
--low 0 \
--high 1 \
--theta 2 \
--obs 10000 \
--marg_flow NSF \
--cop_flow NSF \
--tail_bound_c 64 \
--n_layers_m 5 \
--hidden_units_m 32 \
--n_blocks_m 4 \
--n_bins_m 30 \
--lr_m 0.01 \
--weight_decay_m 1e-06 \
--tail_bound_m 8
