set -e
echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_gaussian_error_nsf_ddsfhp \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal gaussian \
--obs 10000 \
--error_bars \
--cop_flow NSF \
--marg_flow DDSF

