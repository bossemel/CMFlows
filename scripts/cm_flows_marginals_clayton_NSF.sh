set -e
echo 'Begin Copula: Clayton Marginal: uniform Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_Clayton_uniform_nsfnsf \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal_1 uniform \
--marginal_2 uniform \
--alpha 5 \
--low 0 \
--high 1 \
--theta 2 \
--obs 10000 \
--conditional \
--marg_flow NSF \
--cop_flow NSF


echo 'Begin Copula: clayton Marginal: gaussian Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_clayton_gaussian_nsfnsf \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal_1 gaussian \
--marginal_2 gaussian \
--mu 0 \
--var 1 \
--theta 2 \
--obs 10000 \
--conditional \
--marg_flow NSF \
--cop_flow NSF


echo 'Begin Copula: clayton Marginal: gamma Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_clayton_gamma_nsfnsf \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal_1 gamma \
--marginal_2 gamma \
--alpha 5 \
--mu 0 \
--var 1 \
--theta 2 \
--obs 10000 \
--conditional \
--marg_flow NSF \
--cop_flow NSF


echo 'Begin Copula: clayton Marginal: lognormal Pretrain only'

python CM_Flow.py \
--exp_name CM_Flow_clayton_lognormal_nsfnsf \
--epochs 100 \
--batch-size 100 \
--copula clayton \
--marginal_1 lognormal \
--marginal_2 lognormal \
--mu 0 \
--var 1 \
--theta 2 \
--obs 10000 \
--conditional \
--marg_flow NSF \
--cop_flow NSF
