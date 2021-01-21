set -e
echo 'Gamma - random Search'

python NSF.py \
--exp_name NSF_mixlogn_random_search \
--epochs 100 \
--marginal mix_lognormal \
--alpha 5 \
--batch-size 100 \
--obs 10000 \
--random_search \
--flow_type marg_flow
