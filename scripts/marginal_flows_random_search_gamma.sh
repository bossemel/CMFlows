set -e
echo 'Gamma - random Search'

python DDSF.py \
--exp_name DDSF_gamma_random_search \
--marginal gamma \
--obs 10000 \
--random_search \
--continue_from 96
