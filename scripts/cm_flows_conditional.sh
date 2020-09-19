set -e
echo 'Begin Copula: Clayton Marginal: gamma Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_Clayton_gamma_pretr_only_conditional \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal_1 gamma \
--marginal_2 gamma \
--theta 2 \
--obs 10000 \
--pretrain_models \
--conditional_copula

echo 'Begin Copula: frank Marginal: gamma Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_frank_gamma_pretr_only_conditional \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal_1 gamma \
--marginal_2 gamma \
--theta 5 \
--obs 10000 \
--pretrain_models \
--conditional_copula

echo 'Begin Copula: gumbel Marginal: gamma Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_gumbel_gamma_pretr_only_conditional \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal_1 gamma \
--marginal_2 gamma \
--alpha 5 \
--theta 5 \
--obs 10000 \
--pretrain_models \
--conditional_copula
