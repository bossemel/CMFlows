set -e
echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_uniform_error_64 \
--epochs 100 \
--batch-size 128 \
--copula frank \
--marginal uniform \
--obs 10000 \
--error_bars \
--cop_flow NSF \
--marg_flow DDSF \
--tail_bound_c 64

