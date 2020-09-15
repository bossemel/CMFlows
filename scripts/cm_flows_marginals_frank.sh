set -e
# echo 'Begin Copula: frank Marginal: uniform Pretrain only'

# python CM_Flow.py \
# --exp_name CM_Flow_frank_uniform_pretr_only \
# --epochs 100 \
# --batch-size 100 \
# --copula frank \
# --marginal_1 uniform \
# --marginal_2 uniform \
# --alpha 5 \
# --low 0 \
# --high 1 \
# --theta 2 \
# --obs 10000 \
# --pretrain_models

# echo 'Begin Copula: frank Marginal: gaussian Pretrain only'

# python CM_Flow.py \
# --exp_name CM_Flow_frank_gaussian_pretr_only \
# --epochs 100 \
# --batch-size 100 \
# --copula frank \
# --marginal_1 gaussian \
# --marginal_2 gaussian \
# --mu 0 \
# --var 1 \
# --theta 5 \
# --obs 10000 \
# --pretrain_models

# echo 'Begin Copula: frank Marginal: gamma Pretrain only'

# python CM_Flow.py \
# --exp_name CM_Flow_frank_gamma_pretr_only \
# --epochs 100 \
# --batch-size 100 \
# --copula frank \
# --marginal_1 gamma \
# --marginal_2 gamma \
# --alpha 5 \
# --mu 0 \
# --var 1 \
# --theta 5 \
# --obs 10000 \
# --pretrain_models

echo 'Begin Copula: frank Marginal: lognormal Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_frank_lognormal_pretr_only \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal_1 lognormal \
--marginal_2 lognormal \
--mu 0 \
--var 1 \
--theta 5 \
--obs 10000 \
--pretrain_models

echo 'Begin Copula: frank Marginal: bimodal_gaussian Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_frank_bimodal_gaussian_pretr_only \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal_1 bimodal_gaussian \
--marginal_2 bimodal_gaussian \
--mu 0 \
--var 1 \
--theta 5 \
--obs 10000 \
--pretrain_models
