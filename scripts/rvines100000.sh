set -e
echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_100000 \
--epochs 50 \
--batch-size 100 \
--marginal bimodal_gaussian \
--obs 100000 \
--mix

echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_100000 \
--epochs 50 \
--batch-size 100 \
--copula clayton \
--marginal bimodal_gaussian \
--obs 100000

echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_100000 \
--epochs 50 \
--batch-size 100 \
--copula frank \
--marginal bimodal_gaussian \
--obs 100000

echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_100000 \
--epochs 50 \
--batch-size 100 \
--copula gumbel \
--marginal bimodal_gaussian \
--obs 100000
