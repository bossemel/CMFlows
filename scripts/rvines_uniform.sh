set -e
echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_uniform \
--epochs 50 \
--batch-size 100 \
--marginal uniform \
--obs 10000 \
--mix

echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_uniform \
--epochs 50 \
--batch-size 100 \
--copula clayton \
--marginal uniform \
--obs 10000

echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_uniform \
--epochs 50 \
--batch-size 100 \
--copula frank \
--marginal uniform \
--obs 10000

echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_uniform \
--epochs 50 \
--batch-size 100 \
--copula gumbel \
--marginal uniform \
--obs 10000
