set -e
echo 'Begin Copula: kdecopula estimation'

python kdecopula.py \
--exp_name kdecopula_frank_uniform \
--copula frank \
--marginal_1 uniform \
--marginal_2 uniform \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: kdecopula estimation'

python kdecopula.py \
--exp_name kdecopula_frank_gaussian \
--copula frank \
--marginal_1 gaussian \
--marginal_2 gaussian \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: kdecopula estimation'

python kdecopula.py \
--exp_name kdecopula_frank_gamma \
--copula frank \
--marginal_1 gamma \
--marginal_2 gamma \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: kdecopula estimation'

python kdecopula.py \
--exp_name kdecopula_frank_lognormal \
--copula frank \
--marginal_1 lognormal \
--marginal_2 lognormal \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars
