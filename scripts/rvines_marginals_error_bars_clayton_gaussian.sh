set -e
echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_gaussian_error_nsf_ddsfhp \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal gaussian \
--alpha 5 \
--obs 10000 \
--error_bars \
--cop_flow NSF \
--marg_flow DDSF \
--continue_error_bars 8
