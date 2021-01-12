set -e
echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_uniform_error_nsf_ddsfhp \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal uniform \
--obs 10000 \
--error_bars \
--cop_flow NSF \
--marg_flow DDSF \
--continue_error_bars 8


