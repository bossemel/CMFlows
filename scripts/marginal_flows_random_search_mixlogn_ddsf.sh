set -e
echo 'Gamma - random Search'

python DDSF.py \
--exp_name DDSF_mixlogn_random_search \
--epochs 100 \
--marginal mix_lognormal \
--alpha 5 \
--batch-size 128 \
--obs 10000 \
--random_search
