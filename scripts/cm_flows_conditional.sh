set -e
echo 'Begin Copula: Clayton Marginal: uniform Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_Clayton_uniform_pretr_only_conditional \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal_1 uniform \
--marginal_2 uniform \
--theta 2 \
--obs 10000 \
--pretrain_models \
--conditional_copula # \
#--error_bars

# echo 'Begin Copula: frank Marginal: uniform Pretrain only'

# python CM_Flow.py \
# --exp_name CM_Flow_frank_uniform_pretr_only_conditional \
# --epochs 100 \
# --batch-size 100 \
# --copula frank \
# --marginal_1 uniform \
# --marginal_2 uniform \
# --theta 5 \
# --obs 10000 \
# --pretrain_models \
# --conditional_copula \
# --error_bars

# echo 'Begin Copula: gumbel Marginal: uniform Pretrain only'

# python CM_Flow.py \
# --exp_name CM_Flow_gumbel_uniform_pretr_only_conditional \
# --epochs 100 \
# --batch-size 100 \
# --copula gumbel \
# --marginal_1 uniform \
# --marginal_2 uniform \
# --alpha 5 \
# --theta 5 \
# --obs 10000 \
# --pretrain_models \
# --conditional_copula \
# --error_bars
