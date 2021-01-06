set -e
echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_gaussian_error_nsf \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal gaussian \
--obs 10000 \
--error_bars \
--cop_flow NSF \
--marg_flow DDSF
