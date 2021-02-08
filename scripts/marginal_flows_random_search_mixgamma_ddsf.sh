set -e
echo 'Gamma - random Search'

python DDSF.py \
--exp_name DDSF_mixgamma_random_search \
--marginal mix_gamma \
--obs 10000 \
--random_search
