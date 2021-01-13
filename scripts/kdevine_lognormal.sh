set -e
echo 'Begin Mix Copula kdevine: kdevine_mix'

python kdevine.py \
--exp_name kdevine_mix_lognormal \
--marginal lognormal \
--obs 10000 \
--mix \
--error_bars

echo 'Begin Clayton Copula kdevine: kdevine_clayton'

python kdevine.py \
--exp_name kdevine_clayton_lognormal \
--copula clayton \
--marginal lognormal \
--obs 10000 \
--error_bars

echo 'Begin Frank Copula kdevine: kdevine_frank'

python kdevine.py \
--exp_name kdevine_frank_lognormal \
--copula frank \
--marginal lognormal \
--obs 10000 \
--error_bars

echo 'Begin Gumbel Copula kdevine: kdevine_gumbel'

python kdevine.py \
--exp_name kdevine_gumbel_lognormal \
--copula gumbel \
--marginal lognormal \
--obs 10000 \
--error_bars
