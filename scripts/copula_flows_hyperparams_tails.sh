set -e
echo 'Begin Clayton Copula with different Hyperparameters'

python NSF.py \
--exp_name clayton_tails_nsf  \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--theta 2 \
--obs 10000 \
--conditional_copula \
--n_layers_c 15 \
--hidden_units_c 8 \
--n_blocks_c 3 \
--n_bins_c 25 \
--dropout_c 0.15 \
--tail_bound_c 16


python NSF.py \
--exp_name clayton_tails_cm  \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--theta 2 \
--obs 10000 \
--conditional_copula \
--n_layers_c 5 \
--hidden_units_c 64 \
--n_blocks_c 4 \
--n_bins_c 30 \
--dropout_c 0.25 \
--tail_bound_c 32


python NSF.py \
--exp_name clayton_tails_gum_hyp  \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--theta 2 \
--obs 10000 \
--conditional_copula \
--n_layers_c 15 \
--hidden_units_c 8 \
--n_blocks_c 3 \
--n_bins_c 25 \
--dropout_c 0.15 \
--tail_bound_c 16

echo 'Begin Frank Copula with different Hyperparameters'

python NSF.py \
--exp_name frank_tails_nsf  \
--epochs 100 \
--batch-size 100 \
--copula frank \
--theta 2 \
--obs 10000 \
--conditional_copula \
--n_layers_c 15 \
--hidden_units_c 8 \
--n_blocks_c 3 \
--n_bins_c 25 \
--dropout_c 0.15 \
--tail_bound_c 16


python NSF.py \
--exp_name frank_tails_cm  \
--epochs 100 \
--batch-size 100 \
--copula frank \
--theta 2 \
--obs 10000 \
--conditional_copula \
--n_layers_c 5 \
--hidden_units_c 64 \
--n_blocks_c 4 \
--n_bins_c 30 \
--dropout_c 0.25 \
--tail_bound_c 32


python NSF.py \
--exp_name frank_tails_gum_hyp  \
--epochs 100 \
--batch-size 100 \
--copula frank \
--theta 2 \
--obs 10000 \
--conditional_copula \
--n_layers_c 15 \
--hidden_units_c 8 \
--n_blocks_c 3 \
--n_bins_c 25 \
--dropout_c 0.15 \
--tail_bound_c 16

echo 'Begin Gumbel Copula with different Hyperparameters'

python NSF.py \
--exp_name gumbel_tails_nsf  \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--theta 2 \
--obs 10000 \
--conditional_copula \
--n_layers_c 15 \
--hidden_units_c 8 \
--n_blocks_c 3 \
--n_bins_c 25 \
--dropout_c 0.15 \
--tail_bound_c 16


python NSF.py \
--exp_name gumbel_tails_cm  \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--theta 2 \
--obs 10000 \
--conditional_copula \
--n_layers_c 5 \
--hidden_units_c 64 \
--n_blocks_c 4 \
--n_bins_c 30 \
--dropout_c 0.25 \
--tail_bound_c 32


python NSF.py \
--exp_name gumbel_tails_gum_hyp  \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--theta 2 \
--obs 10000 \
--conditional_copula \
--n_layers_c 15 \
--hidden_units_c 8 \
--n_blocks_c 3 \
--n_bins_c 25 \
--dropout_c 0.15 \
--tail_bound_c 16
