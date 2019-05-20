# parser.add_argument('-name', help='run name - will output in this dir', default="Run-"+month+"-"+day)
# parser.add_argument('-KFOLDS', help='Number of folds', default='10')
# parser.add_argument('-FOLD_I', help='This fold i', default='0')
# parser.add_argument('-model_backend', help='Model used in the encoder part of the U-Net structures model', default='resnet50')
# parser.add_argument('-train_epochs', help='How many epochs', default='100')
# parser.add_argument('-train_batch', help='How big batch size', default='8')

#BACKBONE = 'resnet34'
#BACKBONE = 'resnet50' #batch 16
#BACKBONE = 'resnet101' #batch 8

for fold_index in 9
do
   python3 main.py -FOLD_I $fold_index -KFOLDS 10 -model_backend resnet101
done


for fold_index in 0 1 2 3 4 5 6 7 8 9
do
   python3 main.py -FOLD_I $fold_index -KFOLDS 10 -model_backend resnet50 -train_batch 16
done