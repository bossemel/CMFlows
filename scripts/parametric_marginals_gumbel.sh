set -e
echo 'Begin Copula: Parametric estimation'

python parametric.py \
--exp_name parametric_gumbel_uniform \
--copula gumbel \
--assumed_copula gaussian \
--marginal_1 uniform \
--marginal_2 uniform \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: Parametric estimation'

python parametric.py \
--exp_name parametric_gumbel_gaussian \
--copula gumbel \
--assumed_copula gaussian \
--marginal_1 gaussian \
--marginal_2 gaussian \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: Parametric estimation'

python parametric.py \
--exp_name parametric_gumbel_gamma \
--copula gumbel \
--assumed_copula gaussian \
--marginal_1 gamma \
--marginal_2 gamma \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: Parametric estimation'

python parametric.py \
--exp_name parametric_gumbel_lognormal \
--copula gumbel \
--assumed_copula gaussian \
--marginal_1 lognormal \
--marginal_2 lognormal \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: Parametric estimation'

python parametric.py \
--exp_name parametric_gumbel_bimodal_gaussian \
--copula gumbel \
--assumed_copula gaussian \
--marginal_1 bimodal_gaussian \
--marginal_2 bimodal_gaussian \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars
