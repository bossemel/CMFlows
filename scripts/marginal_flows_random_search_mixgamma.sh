set -e
echo 'Gamma - random Search'

python NSF.py \
--exp_name NSF_mixgamma_random_search \
--epochs 100 \
--marginal mix_gamma \
--alpha 5 \
--batch-size 128 \
--obs 10000 \
--random_search \
--flow_type marg_flow
