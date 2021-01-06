set -e
echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_gamma_error_nsf \
--epochs 100 \
--batch-size 100 \
--marginal gamma \
--obs 10000 \
--mix \
--error_bars \
--cop_flow NSF \
--marg_flow DDSF

