set -e
echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_uniform_error_64 \
--epochs 100 \
--batch-size 100 \
--marginal uniform \
--obs 10000 \
--mix \
--error_bars \
--cop_flow NSF \
--marg_flow DDSF \
--tail_bound_c 64
