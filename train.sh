python ./train/classification.py --mode clear --data_name geo --cls_model resnet --device cuda:0 --batchsize1 32

python ./train/make_cam.py 

# cd train/
# python segmentation/mainNet.py --mode clear --data_name geo --device cuda:0 --learning_rate 0.0001
# python segmentation/predict.py --mode clear --data_name geo --device cuda:0

# cd ..