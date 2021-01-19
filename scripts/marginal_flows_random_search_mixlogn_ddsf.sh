set -e
echo 'Gamma - random Search'

python DDSF.py \
--exp_name DDSF_gmm_random_search \
--epochs 100 \
--marginal mix_gamma \
--alpha 5 \
--batch-size 100 \
--obs 10000 \
--random_search
