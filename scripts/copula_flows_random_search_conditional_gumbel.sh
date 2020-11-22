set -e
# echo 'Begin Clayton Copula Grid Search'

# python NSF.py \
# --exp_name clayton_random_search \
# --epochs 100 \
# --copula clayton \
# --theta 2 \
# --batch-size 100 \
# --obs 10000 \
# --random_search \
# --conditional_copula \
# --flow_type cop_flow

# echo 'Begin Frank Copula random Search'

# python NSF.py \
# --exp_name frank_random_search \
# --epochs 100 \
# --copula frank \
# --theta 5 \
# --batch-size 100 \
# --obs 10000 \
# --random_search \
# --conditional_copula \
# --flow_type cop_flow

echo 'Begin Gumbel Copula random Search'

python NSF.py \
--exp_name gumbel_random_search \
--epochs 100 \
--copula gumbel \
--theta 5 \
--batch-size 100 \
--obs 10000 \
--random_search \
--conditional_copula \
--flow_type cop_flow \
--continue_from 110
