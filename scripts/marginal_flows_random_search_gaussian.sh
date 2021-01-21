set -e
echo 'Gaussian - random Search'

python NSF.py \
--exp_name NSF_gaussian_random_search \
--marginal gaussian \
--obs 10000 \
--random_search \
--flow_type marg_flow
