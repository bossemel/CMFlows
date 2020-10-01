set -e


# echo 'Begin Clayton Copula Rvine: RVine_clayton'

# python RVine.py \
# --exp_name RVine_clayton_gaussian_error_new \
# --epochs 100 \
# --batch-size 100 \
# --copula clayton \
# --marginal gaussian \
# --alpha 5 \
# --obs 10000 \
# --error_bars

echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_gamma_error_4_256 \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal gamma \
--alpha 5 \
--obs 10000 \
--error_bars
