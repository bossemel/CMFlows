set -e
echo 'Uniform - random Search'

python NSF.py \
--exp_name NSF_uniform_random_search \
--epochs 100 \
--marginal uniform \
--low -1 \
--high 3 \
--batch-size 128 \
--obs 10000 \
--random_search \
--flow_type marg_flow
