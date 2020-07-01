set -e
echo 'Bimodal Gaussian - Grid Search'

python DDSF.py \
--exp_name DDSF_bimodal_gaussian_grid_search \
--epochs 50 \
--marginal bimodal_gaussian \
--batch-size 100 \
--obs 10000 \
--grid_search

echo 'Gaussian - Grid Search'

python DDSF.py \
--exp_name DDSF_gaussian_grid_search \
--epochs 50 \
--marginal gaussian \
--mu -2 \
--var 3 \
--batch-size 100 \
--obs 10000 \
--grid_search

echo 'Uniform - Grid Search'

python DDSF.py \
--exp_name DDSF_uniform_grid_search \
--epochs 50 \
--marginal uniform \
--low -1 \
--high 3 \
--batch-size 100 \
--obs 10000 \
--grid_search

echo 'Gamma - Grid Search'

python DDSF.py \
--exp_name DDSF_gamma_grid_search \
--epochs 50 \
--marginal gamma \
--alpha 5 \
--batch-size 100 \
--obs 10000 \
--grid_search

echo 'Lognormal - Grid Search'

python DDSF.py \
--exp_name DDSF_lognormal_grid_search \
--epochs 50 \
--marginal lognormal \
--mu 0 \
--var 1 \
--batch-size 100 \
--obs 10000 \
--grid_search
