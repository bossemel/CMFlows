set -e
echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_gaussian_error_64 \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal gaussian \
--alpha 5 \
--obs 10000 \
--error_bars \
--cop_flow NSF \
--marg_flow DDSF \
--tail_bound_c 64
