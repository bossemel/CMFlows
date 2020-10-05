set -e

# echo 'Begin Mix Copula Rvine: RVine_mix'

# python RVine.py \
# --exp_name RVine_gumbel_uniform_marginal_nsf \
# --epochs 50 \
# --batch-size 100 \
# --marginal uniform \
# --copula gumbel \
# --mu 0 \
# --var 1 \
# --obs 10000 \
# --random_seed 3


# echo 'Begin Mix Copula Rvine: RVine_mix'

# python RVine.py \
# --exp_name RVine_mix_gaussian_nsf \
# --epochs 100 \
# --batch-size 100 \
# --marginal gamma \
# --mu 0 \
# --var 1 \
# --obs 10000 \
# --mix

# echo 'Begin Mix Copula Rvine: RVine_mix'

# python RVine.py \
# --exp_name RVine_mix_gamma_nsf \
# --epochs 100 \
# --batch-size 100 \
# --marginal gamma \
# --mu 0 \
# --var 1 \
# --obs 10000 \
# --mix

# echo 'Begin Clayton Copula Rvine: RVine_clayton'

# python RVine.py \
# --exp_name RVine_clayton_gaussian_nsf \
# --epochs 100 \
# --batch-size 100 \
# --copula clayton \
# --marginal gaussian \
# --alpha 5 \
# --obs 10000

# echo 'Begin Clayton Copula Rvine: RVine_clayton'

# python RVine.py \
# --exp_name RVine_clayton_gamma_nsf \
# --epochs 100 \
# --batch-size 100 \
# --copula clayton \
# --marginal gamma \
# --alpha 5 \
# --obs 10000

# echo 'Begin Frank Copula Rvine: RVine_frank'

# python RVine.py \
# --exp_name RVine_frank_gaussian_nsf \
# --epochs 100 \
# --batch-size 100 \
# --copula frank \
# --marginal gaussian \
# --mu 0 \
# --var 1 \
# --obs 10000

# echo 'Begin Frank Copula Rvine: RVine_frank'

# python RVine.py \
# --exp_name RVine_frank_gamma_nsf \
# --epochs 100 \
# --batch-size 100 \
# --copula frank \
# --marginal gamma \
# --mu 0 \
# --var 1 \
# --obs 10000

echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_gaussian_nsf \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal gaussian \
--obs 10000

# echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

# python RVine.py \
# --exp_name RVine_gumbel_gamma_nsf \
# --epochs 100 \
# --batch-size 100 \
# --copula gumbel \
# --marginal gamma \
# --obs 10000
