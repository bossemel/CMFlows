set -e
echo 'Gaussian - random Search'

python NSF.py \
--exp_name NSF_gaussian_random_search \
--epochs 100 \
--marginal gaussian \
--batch-size 100 \
--obs 10000 \
--random_search \
--flow_type marg_flow

# echo 'Uniform - random Search'

# python NSF.py \
# --exp_name NSF_uniform_random_search \
# --epochs 100 \
# --marginal uniform \
# --low -1 \
# --high 3 \
# --batch-size 100 \
# --obs 10000 \
# --random_search \
# --flow_type marg_flow

# echo 'Gamma - random Search'

# python NSF.py \
# --exp_name NSF_gamma_random_search \
# --epochs 100 \
# --marginal gamma \
# --alpha 5 \
# --batch-size 100 \
# --obs 10000 \
# --random_search \
# --flow_type marg_flow

# echo 'Lognormal - random Search'

# python NSF.py \
# --exp_name NSF_lognormal_random_search \
# --epochs 100 \
# --marginal lognormal \
# --mu 0 \
# --var 1 \
# --batch-size 100 \
# --obs 10000 \
# --random_search \
# --flow_type marg_flow
