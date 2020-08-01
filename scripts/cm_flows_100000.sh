set -e
echo 'Begin Copula: Clayton Marginal: bimodal_gaussian Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_Clayton_bimodal_gaussian_pretr_only_100000 \
--epochs 50 \
--batch-size 100 \
--copula clayton \
--marginal_1 bimodal_gaussian \
--marginal_2 bimodal_gaussian \
--theta 2 \
--obs 100000 \
--pretrain_models

echo 'Begin Copula: frank Marginal: bimodal_gaussian Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_frank_bimodal_gaussian_pretr_only_100000 \
--epochs 50 \
--batch-size 100 \
--copula frank \
--marginal_1 bimodal_gaussian \
--marginal_2 bimodal_gaussian \
--theta 5 \
--obs 100000 \
--pretrain_models

echo 'Begin Copula: gumbel Marginal: bimodal_gaussian Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_gumbel_bimodal_gaussian_pretr_only_100000 \
--epochs 50 \
--batch-size 100 \
--copula gumbel \
--marginal_1 bimodal_gaussian \
--marginal_2 bimodal_gaussian \
--theta 5 \
--obs 100000 \
--pretrain_models
