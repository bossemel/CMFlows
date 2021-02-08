set -e
echo 'Gamma - random Search'

python DDSF.py \
--exp_name DDSF_gmm_random_search \
--marginal gmm \
--batch-size 128 \
--obs 10000 \
--random_search
