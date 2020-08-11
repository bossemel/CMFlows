set -e
echo 'Bimodal Gaussian'

python DDSF.py \
--exp_name DDSF_bimodal_gaussian_besthyp \
--epochs 50 \
--marginal bimodal_gaussian \
--batch-size 100 \
--obs 10000

echo 'Gaussian'

python DDSF.py \
--exp_name DDSF_gaussian_besthyp \
--epochs 100 \
--marginal gaussian \
--mu -2 \
--var 3 \
--batch-size 100 \
--obs 10000

echo 'Uniform'

python DDSF.py \
--exp_name DDSF_uniform_besthyp \
--epochs 50 \
--marginal uniform \
--low -1 \
--high 3 \
--batch-size 100 \
--obs 10000

echo 'Gamma'

python DDSF.py \
--exp_name DDSF_gamma_besthyp \
--epochs 50 \
--marginal gamma \
--alpha 5 \
--batch-size 100 \
--obs 10000

echo 'Lognormal'

python DDSF.py \
--exp_name DDSF_lognormal_besthyp \
--epochs 50 \
--marginal lognormal \
--mu 0 \
--var 1 \
--batch-size 100 \
--obs 10000
