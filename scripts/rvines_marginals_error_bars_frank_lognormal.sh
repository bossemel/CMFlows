set -e

# echo 'Begin Frank Copula Rvine: RVine_frank'

# python RVine.py \
# --exp_name RVine_frank_gaussian_error_new \
# --epochs 100 \
# --batch-size 100 \
# --copula frank \
# --marginal gaussian \
# --mu 0 \
# --var 1 \
# --obs 10000 \
# --error_bars \
# --continue_error_bars 8

# echo 'Begin Frank Copula Rvine: RVine_frank'

# python RVine.py \
# --exp_name RVine_frank_gamma_error_new \
# --epochs 100 \
# --batch-size 100 \
# --copula frank \
# --marginal gamma \
# --mu 0 \
# --var 1 \
# --obs 10000 \
# --error_bars

echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_lognormal_error_4_256 \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal lognormal \
--mu 0 \
--var 1 \
--obs 10000 \
--error_bars
