set -e
echo 'Begin Copula: gumbel Marginal: uniform Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_gumbel_uniform_nsf \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal_1 uniform \
--marginal_2 uniform \
--alpha 5 \
--low 0 \
--high 1 \
--theta 2 \
--obs 10000 \
--conditional

echo 'Begin Copula: gumbel Marginal: gaussian Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_gumbel_gaussian_nsf \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal_1 gaussian \
--marginal_2 gaussian \
--mu 0 \
--var 1 \
--theta 5 \
--obs 10000 \
--conditional

echo 'Begin Copula: gumbel Marginal: gamma Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_gumbel_gamma_nsf \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal_1 gamma \
--marginal_2 gamma \
--alpha 5 \
--mu 0 \
--var 1 \
--theta 5 \
--obs 10000 \
--conditional

echo 'Begin Copula: gumbel Marginal: lognormal Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_gumbel_lognormal_nsf \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal_1 lognormal \
--marginal_2 lognormal \
--mu 0 \
--var 1 \
--theta 5 \
--obs 10000 \
--conditional
