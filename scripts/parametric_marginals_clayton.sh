set -e
echo 'Begin Copula: Parametric estimation'

python parametric.py \
--exp_name parametric_clayton_uniform \
--copula clayton \
--assumed_copula gaussian \
--marginal_1 uniform \
--marginal_2 uniform \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: Parametric estimation'

python parametric.py \
--exp_name parametric_clayton_gaussian \
--copula clayton \
--assumed_copula gaussian \
--marginal_1 gaussian \
--marginal_2 gaussian \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: Parametric estimation'

python parametric.py \
--exp_name parametric_clayton_gamma \
--copula clayton \
--assumed_copula gaussian \
--marginal_1 gamma \
--marginal_2 gamma \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: Parametric estimation'

python parametric.py \
--exp_name parametric_clayton_lognormal \
--copula clayton \
--assumed_copula gaussian \
--marginal_1 lognormal \
--marginal_2 lognormal \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: Parametric estimation'

python parametric.py \
--exp_name parametric_clayton_bimodal_gaussian \
--copula clayton \
--assumed_copula gaussian \
--marginal_1 bimodal_gaussian \
--marginal_2 bimodal_gaussian \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars
