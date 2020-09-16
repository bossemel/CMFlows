set -e
echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_uniform \
--epochs 100 \
--batch-size 100 \
--marginal uniform \
--mu 0 \
--var 1 \
--obs 10000 \
--mix

echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_gaussian \
--epochs 100 \
--batch-size 100 \
--marginal gaussian \
--mu 0 \
--var 1 \
--obs 10000 \
--mix

echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_gamma \
--epochs 100 \
--batch-size 100 \
--marginal gamma \
--mu 0 \
--var 1 \
--obs 10000 \
--mix

echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_lognormal \
--epochs 100 \
--batch-size 100 \
--marginal lognormal \
--mu 0 \
--var 1 \
--obs 10000 \
--mix

echo 'Begin Mix Copula Rvine: RVine_mix'

python RVine.py \
--exp_name RVine_mix_bimodal_gaussian \
--epochs 100 \
--batch-size 100 \
--marginal bimodal_gaussian \
--mu 0 \
--var 1 \
--obs 10000 \
--mix

echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_uniform \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal uniform \
--alpha 5 \
--obs 10000

echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_gaussian \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal gaussian \
--alpha 5 \
--obs 10000

echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_gamma \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal gamma \
--alpha 5 \
--obs 10000

echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_lognormal \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal lognormal \
--alpha 5 \
--obs 10000

echo 'Begin Clayton Copula Rvine: RVine_clayton'

python RVine.py \
--exp_name RVine_clayton_bimodal_gaussian \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal bimodal_gaussian \
--alpha 5 \
--obs 10000

echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_uniform \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal uniform \
--mu 0 \
--var 1 \
--obs 10000

echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_gaussian \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal gaussian \
--mu 0 \
--var 1 \
--obs 10000

echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_gamma \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal gamma \
--mu 0 \
--var 1 \
--obs 10000

echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_lognormal \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal lognormal \
--mu 0 \
--var 1 \
--obs 10000

echo 'Begin Frank Copula Rvine: RVine_frank'

python RVine.py \
--exp_name RVine_frank_bimodal_gaussian \
--epochs 100 \
--batch-size 100 \
--copula frank \
--marginal bimodal_gaussian \
--mu 0 \
--var 1 \
--obs 10000

echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_uniform \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal uniform \
--low 0 \
--high 1 \
--obs 10000

echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_gaussian \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal gaussian \
--low 0 \
--high 1 \
--obs 10000

echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_gamma \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal gamma \
--low 0 \
--high 1 \
--obs 10000

echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_lognormal \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal lognormal \
--low 0 \
--high 1 \
--obs 10000

echo 'Begin Gumbel Copula Rvine: RVine_gumbel'

python RVine.py \
--exp_name RVine_gumbel_bimodal_gaussian \
--epochs 100 \
--batch-size 100 \
--copula gumbel \
--marginal bimodal_gaussian \
--low 0 \
--high 1 \
--obs 10000
