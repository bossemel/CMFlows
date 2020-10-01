set -e
echo 'Begin Mix Copula Rvine: kdevine_mix'

python kdevine.py \
--exp_name kdevine_mix_gaussian_100000 \
--marginal gamma \
--mu 0 \
--var 1 \
--obs 100000 \
--mix

echo 'Begin Mix Copula Rvine: kdevine_mix'

python kdevine.py \
--exp_name kdevine_mix_gamma_100000 \
--marginal gamma \
--mu 0 \
--var 1 \
--obs 100000 \
--mix

echo 'Begin Clayton Copula Rvine: kdevine_clayton'

python kdevine.py \
--exp_name kdevine_clayton_gaussian_100000 \
--copula clayton \
--marginal gaussian \
--alpha 5 \
--obs 100000

echo 'Begin Clayton Copula Rvine: kdevine_clayton'

python kdevine.py \
--exp_name kdevine_clayton_gamma_100000 \
--copula clayton \
--marginal gamma \
--alpha 5 \
--obs 100000

echo 'Begin Frank Copula Rvine: kdevine_frank'

python kdevine.py \
--exp_name kdevine_frank_gaussian_100000 \
--copula frank \
--marginal gaussian \
--mu 0 \
--var 1 \
--obs 100000

echo 'Begin Frank Copula Rvine: kdevine_frank'

python kdevine.py \
--exp_name kdevine_frank_gamma_100000 \
--copula frank \
--marginal gamma \
--mu 0 \
--var 1 \
--obs 100000

echo 'Begin Gumbel Copula Rvine: kdevine_gumbel'

python kdevine.py \
--exp_name kdevine_gumbel_gaussian_100000 \
--copula gumbel \
--marginal gaussian \
--obs 100000

echo 'Begin Gumbel Copula Rvine: kdevine_gumbel'

python kdevine.py \
--exp_name kdevine_gumbel_gamma_100000 \
--copula gumbel \
--marginal gamma \
--obs 100000
