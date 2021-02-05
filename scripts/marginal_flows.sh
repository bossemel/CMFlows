set -e
echo 'Gaussian'

python DDSF.py \
--exp_name DDSF_gaussian_128 \
--epochs 100 \
--marginal gaussian \
--mu -2 \
--var 3 \
--batch-size 128 \
--obs 10000 \
--random_seed 4

echo 'Uniform'

python DDSF.py \
--exp_name DDSF_uniform_128 \
--epochs 100 \
--marginal uniform \
--low -1 \
--high 3 \
--batch-size 128 \
--obs 10000 \
--random_seed 4

echo 'Gamma'

python DDSF.py \
--exp_name DDSF_gamma_128 \
--epochs 100 \
--marginal gamma \
--alpha 5 \
--batch-size 128 \
--obs 10000 \
--random_seed 4

echo 'Lognormal'

python DDSF.py \
--exp_name DDSF_lognormal_128 \
--epochs 100 \
--marginal lognormal \
--mu 0 \
--var 1 \
--batch-size 128 \
--obs 10000 \
--random_seed 4
