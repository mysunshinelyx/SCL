# SCL
Y. Liu, J. Wu, L. Qu, T. Gan, J. Yin and L. Nie, "Self-supervised Correlation Learning for Cross-Modal Retrieval," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2022.3152086. 
# Datasets and Features 
The features of the wikipedia and nus-wide-10k datasets were provided by https://github.com/sunpeng981712364/ACMR_demo.
# Usage
## Unsupervised Setting
python main.py --datasets wikipedia --lr 0.0001 --alpha 100 --beta 0.01 --gamma 1
## Semi-supervised setting
python main.py --datasets wikipedia --semi_set 1 --sup_rate 0.1 --lr 0.01 --alpha 10 --beta 1 --gamma 1 --delta 10 --theta 1
