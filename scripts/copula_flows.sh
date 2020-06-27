set -e
echo 'Begin Clayton Copula Grid Search'

python RealNVP.py \
--exp_name clayton_grid_search \
--epochs 100 \
--copula clayton \
--theta 2 \
--batch-size 1000 \
--obs 10000 \
--random_seed 58093 \
--grid_search \
--early_stopping

set -e

echo 'Begin Frank Copula Grid Search'

python RealNVP.py \
--exp_name frank_grid_search \
--epochs 100 \
--copula frank \
--theta 2 \
--batch-size 1000 \
--obs 10000 \
--random_seed 58093 \
--grid_search \
--early_stopping

echo 'Begin Gumbel Copula Grid Search'

python RealNVP.py \
--exp_name gumbel_grid_search \
--epochs 100 \
--copula gumbel \
--theta 2 \
--batch-size 1000 \
--obs 10000 \
--random_seed 58093 \
--grid_search \
--early_stopping

# stop

# echo 'Begin frank Copula Experiment - Sigmoid'

# python RealNVP.py \
# --exp_name frank_10000_100_sigmoid \
# --epochs 100 \
# --copula frank \
# --theta 5 \
# --batch-size 1000 \
# # --num-blocks 5 \
# --obs 10000 \
# --num_hidden 64 \
# --random_seed 58093 \
# --transform_fct sigmoid

# echo 'Begin gumbel Copula Experiment - Sigmoid'

# python RealNVP.py \
# --exp_name gumbel_10000_100_sigmoid \
# --epochs 100 \
# --copula gumbel \
# --theta 5 \
# --batch-size 1000 \
# # --num-blocks 5 \
# --obs 10000 \
# --num_hidden 64 \
# --random_seed 58093 \
# --transform_fct sigmoid

# echo 'Begin clayton Copula Experiment - Gaussian'

# python RealNVP.py \
# --exp_name clayton_10000_100_gaussian \
# --epochs 100 \
# --copula clayton \
# --theta 2 \
# --batch-size 1000 \
# # --num-blocks 5 \
# --obs 10000 \
# --num_hidden 64 \
# --random_seed 58093 \
# --transform_fct gaussian

# echo 'Begin frank Copula Experiment - gaussian'

# python RealNVP.py \
# --exp_name frank_10000_100_gaussian \
# --epochs 100 \
# --copula frank \
# --theta 5 \
# --batch-size 1000 \
# # --num-blocks 5 \
# --obs 10000 \
# --num_hidden 64 \
# --random_seed 58093 \
# --transform_fct gaussian

# echo 'Begin gumbel Copula Experiment - gaussian'

# python RealNVP.py \
# --exp_name gumbel_10000_100_gaussian \
# --epochs 100 \
# --copula gumbel \
# --theta 5 \
# --batch-size 1000 \
# # --num-blocks 5 \
# --obs 10000 \
# --num_hidden 64 \
# --random_seed 58093 \
# --transform_fct gaussian
