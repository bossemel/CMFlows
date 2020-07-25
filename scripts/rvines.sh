set -e
echo 'Begin Copula: Clayton Marginal: bimodal_gaussian Pretrain only'

python RVine.py \
--epochs 10 \
--batch-size 100 \
--copula clayton \
--marginal uniform \
--obs 10000 \
--pretrain_models
