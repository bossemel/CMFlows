set -e

echo 'Gaussian'

python NSF.py \
--exp_name NSF_gaussian_besthyp \
--epochs 100 \
--marginal gaussian \
--mu -2 \
--var 3 \
--batch-size 100 \
--obs 10000 \
--flow_type marg_flow

echo 'Uniform'

python NSF.py \
--exp_name NSF_uniform_besthyp \
--epochs 100 \
--marginal uniform \
--low -1 \
--high 3 \
--batch-size 100 \
--obs 10000 \
--flow_type marg_flow

echo 'Gamma'

python NSF.py \
--exp_name NSF_gamma_besthyp \
--epochs 100 \
--marginal gamma \
--alpha 5 \
--batch-size 100 \
--obs 10000 \
--flow_type marg_flow

echo 'Lognormal'

python NSF.py \
--exp_name NSF_lognormal_besthyp \
--epochs 100 \
--marginal lognormal \
--mu 0 \
--var 1 \
--batch-size 100 \
--obs 10000 \
--flow_type marg_flow
