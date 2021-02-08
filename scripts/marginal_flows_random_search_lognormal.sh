set -e
echo 'Lognormal - random Search'

python NSF.py \
--exp_name NSF_lognormal_random_search \
--epochs 100 \
--marginal lognormal \
--mu 0 \
--var 1 \
--batch-size 128 \
--obs 10000 \
--random_search \
--flow_type marg_flow
