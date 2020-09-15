set -e
echo 'Begin Copula: Parametric estimation'

python parametric.py \
--exp_name parametric_clayton \
--copula clayton \
--assumed_copula gumbel \
--marginal_1 gamma \
--marginal_2 gamma \
--alpha 5 \
--theta 2 \
--obs 10000 \
--error_bars

echo 'Begin Copula: frank Marginal: bimodal_gaussian Pretrain only'

python parametric.py \
--exp_name parametric_frank \
--copula frank \
--assumed_copula clayton \
--marginal_1 gamma \
--marginal_2 gamma \
--mu 0 \
--var 1 \
--theta 5 \
--obs 10000 \
--error_bars

echo 'Begin Copula: gumbel Marginal: bimodal_gaussian Pretrain only'

python parametric.py \
--exp_name parametric_gumbel \
--copula gumbel \
--assumed_copula clayton \
--marginal_1 gamma \
--marginal_2 gamma \
--mu 0 \
--var 1 \
--theta 5 \
--obs 10000 \
--error_bars
