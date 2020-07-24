set -e
echo 'Begin Copula: Clayton Marginal: bimodal_gaussian Pretrain only'

python RVine.py \
--epochs 1 \
--batch-size 100 \
--copula clayton \
--marginal_1 bimodal_gaussian \
--marginal_2 bimodal_gaussian \
--theta 2 \
--obs 10000 \
--pretrain_models
