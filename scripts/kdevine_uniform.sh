set -e
echo 'Begin Mix Copula kdevine: kdevine_mix'

python kdevine.py \
--exp_name kdevine_mix_uniform \
--marginal uniform \
--obs 10000 \
--mix \
--error_bars

echo 'Begin Clayton Copula kdevine: kdevine_clayton'

python kdevine.py \
--exp_name kdevine_clayton_uniform \
--copula clayton \
--marginal uniform \
--obs 10000 \
--error_bars

echo 'Begin Frank Copula kdevine: kdevine_frank'

python kdevine.py \
--exp_name kdevine_frank_uniform \
--copula frank \
--marginal uniform \
--obs 10000 \
--error_bars

echo 'Begin Gumbel Copula kdevine: kdevine_gumbel'

python kdevine.py \
--exp_name kdevine_gumbel_uniform \
--copula gumbel \
--marginal uniform \
--obs 10000 \
--error_bars
