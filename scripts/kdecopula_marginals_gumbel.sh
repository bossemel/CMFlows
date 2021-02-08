set -e
# echo 'Begin Copula: kdecopula estimation'

# python kdecopula.py \
# --exp_name kdecopula_gumbel_uniform \
# --copula gumbel \
# --marginal_1 uniform \
# --marginal_2 uniform \
# --alpha 5 \
# --theta 2 \
# --obs 10000 \
# --error_bars

# echo 'Begin Copula: kdecopula estimation'

# python kdecopula.py \
# --exp_name kdecopula_gumbel_gaussian \
# --copula gumbel \
# --marginal_1 gaussian \
# --marginal_2 gaussian \
# --alpha 5 \
# --theta 2 \
# --obs 10000 \
# --error_bars

# echo 'Begin Copula: kdecopula estimation'

# python kdecopula.py \
# --exp_name kdecopula_gumbel_gamma \
# --copula gumbel \
# --marginal_1 gamma \
# --marginal_2 gamma \
# --alpha 5 \
# --theta 2 \
# --obs 10000 \
# --error_bars

# echo 'Begin Copula: kdecopula estimation'

# python kdecopula.py \
# --exp_name kdecopula_gumbel_lognormal \
# --copula gumbel \
# --marginal_1 lognormal \
# --marginal_2 lognormal \
# --alpha 5 \
# --theta 2 \
# --obs 10000 \
# --error_bars



echo 'Begin Copula: kdecopula estimation'

python kdecopula.py \
--exp_name kdecopula_gumbel_gmm \
--copula gumbel \
--marginal_1 gmm \
--marginal_2 gmm \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: kdecopula estimation'

python kdecopula.py \
--exp_name kdecopula_gumbel_mix_gamma \
--copula gumbel \
--marginal_1 mix_gamma \
--marginal_2 mix_gamma \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: kdecopula estimation'

python kdecopula.py \
--exp_name kdecopula_gumbel_mix_lognormal \
--copula gumbel \
--marginal_1 mix_lognormal \
--marginal_2 mix_lognormal \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars
