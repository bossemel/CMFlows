set -e
echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_lognormal \
--epochs 1 \
--batch-size 100 \
--marginal lognormal \
--mu 0 \
--var 1 \
--obs 10000 \
--mix

echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_lognormal \
--epochs 1 \
--batch-size 100 \
--copula clayton \
--marginal lognormal \
--alpha 5 \
--obs 10000

echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_lognormal \
--epochs 1 \
--batch-size 100 \
--copula frank \
--marginal lognormal \
--mu 0 \
--var 1 \
--obs 10000

echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_lognormal \
--epochs 1 \
--batch-size 100 \
--copula gumbel \
--marginal lognormal \
--low 0 \
--high 1 \
--obs 10000
