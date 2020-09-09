set -e
echo 'Bimodal Gaussian - random Search'

python DDSF.py \
--exp_name DDSF_bimodal_gaussian_random_search \
--epochs 50 \
--marginal bimodal_gaussian \
--batch-size 100 \
--obs 10000 \
--random_search \
--early_stopping

# echo 'Gaussian - random Search'

# python DDSF.py \
# --exp_name DDSF_gaussian_random_search \
# --epochs 50 \
# --marginal gaussian \
# --mu -2 \
# --var 3 \
# --batch-size 100 \
# --obs 10000 \
# --random_search \
# --early_stopping

# echo 'Uniform - random Search'

# python DDSF.py \
# --exp_name DDSF_uniform_random_search \
# --epochs 50 \
# --marginal uniform \
# --low -1 \
# --high 3 \
# --batch-size 100 \
# --obs 10000 \
# --random_search \
# --early_stopping

# echo 'Gamma - random Search'

# python DDSF.py \
# --exp_name DDSF_gamma_random_search \
# --epochs 50 \
# --marginal gamma \
# --alpha 5 \
# --batch-size 100 \
# --obs 10000 \
# --random_search \
# --early_stopping

# echo 'Lognormal - random Search'

# python DDSF.py \
# --exp_name DDSF_lognormal_random_search \
# --epochs 50 \
# --marginal lognormal \
# --mu 0 \
# --var 1 \
# --batch-size 100 \
# --obs 10000 \
# --random_search \
# --early_stopping
