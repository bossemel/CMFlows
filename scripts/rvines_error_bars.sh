set -e
echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_1_step \
--epochs 50 \
--batch-size 100 \
--marginal bimodal_gaussian \
--obs 10000 \
--mix \
--error_bars

echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_1_step \
--epochs 50 \
--batch-size 100 \
--copula clayton \
--marginal bimodal_gaussian \
--obs 10000 \
--error_bars

echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_1_step \
--epochs 50 \
--batch-size 100 \
--copula frank \
--marginal bimodal_gaussian \
--obs 10000 \
--error_bars

echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_1_step \
--epochs 50 \
--batch-size 100 \
--copula gumbel \
--marginal bimodal_gaussian \
--obs 10000 \
--error_bars
