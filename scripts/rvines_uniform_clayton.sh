set -e


echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_uniform \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal uniform \
--obs 10000 \
--disable_marginal \
--error_bars
