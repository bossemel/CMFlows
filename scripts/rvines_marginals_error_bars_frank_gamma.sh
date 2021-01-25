set -e
echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_gamma_error_64 \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal gamma \
--obs 10000 \
--error_bars \
--cop_flow NSF \
--marg_flow DDSF \
--tail_bound_c 64 \
--use_ecdf

