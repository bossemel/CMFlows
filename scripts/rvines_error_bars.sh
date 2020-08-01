set -e
# echo 'Begin Mix Copula Rvine: RVine_mix'

# python RVine.py \
# --exp_name RVine_mix \
# --epochs 1 \
# --batch-size 100 \
# --marginal bimodal_gaussian \
# --obs 10000 \
# --pretrain_models \
# --mix \
# --error_bars

echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton \
--epochs 1 \
--batch-size 100 \
--copula clayton \
--marginal bimodal_gaussian \
--obs 10000 \
--pretrain_models \
--error_bars

echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank \
--epochs 1 \
--batch-size 100 \
--copula frank \
--marginal bimodal_gaussian \
--obs 10000 \
--pretrain_models \
--error_bars

echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel \
--epochs 1 \
--batch-size 100 \
--copula gumbel \
--marginal bimodal_gaussian \
--obs 10000 \
--pretrain_models \
--error_bars
