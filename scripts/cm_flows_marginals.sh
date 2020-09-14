set -e
echo 'Begin Copula: Clayton Marginal: bimodal_gaussian Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_Clayton_uniform_pretr_only \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal_1 uniform \
--marginal_2 uniform \
--low 0 \
--theta 2 \
--obs 10000 \
--pretrain_models

echo 'Begin Copula: frank Marginal: bimodal_gaussian Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_frank_gaussian_pretr_only \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal_1 gaussian \
--marginal_2 gaussian \
--theta 5 \
--obs 10000 \
--pretrain_models

echo 'Begin Copula: gumbel Marginal: bimodal_gaussian Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_gumbel_bimodal_gaussian_pretr_only \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal_1 lognormal \
--marginal_2 lognormal \
--alpha 5 \
--theta 5 \
--obs 10000 \
--pretrain_models

echo 'Begin Copula: gam Marginal: bimodal_gaussian Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_gumbel_gamma_pretr_only \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal_1 gamma \
--marginal_2 gamma \
--alpha 5 \
--theta 5 \
--obs 10000 \
--pretrain_models
