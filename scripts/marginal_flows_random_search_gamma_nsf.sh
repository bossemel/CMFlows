set -e
echo 'Gamma - random Search'

python NSF.py \
--exp_name NSF_gamma_random_search \
--marginal gamma \
--obs 10000 \
--random_search \
--flow_type marg_flow
