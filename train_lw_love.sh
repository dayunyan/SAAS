MDOE=clear
DATANAME=love
CUDA=cuda:0
CAM=xgradcam
NET=fsan
WO=none

cd train/
# python classification_lw_woDARM.py --mode clear --data_name $dataname --cls_model resnet --device $CUDA --batchsize1 8
# python classification_lw_con.py --mode $MDOE --data_name $DATANAME --cls_model resnet --device $CUDA --batchsize1 4 --cam $CAM
# python classification_lw_love.py --mode $MDOE --data_name $DATANAME --cls_model resnet --device $CUDA --batchsize1 32 --cam $CAM

python make_cam_lw_love.py --mode $MDOE --data_name $DATANAME --device $CUDA --cam $CAM

# python segmentation/mainNet_fsan.py --mode $MDOE --data_name $DATANAME --device $CUDA --learning_rate 0.0005 --cam $CAM --net $NET --wo $WO
# python segmentation/predict_fsan.py --mode $MDOE --data_name $DATANAME --device $CUDA --cam $CAM --net $NET --wo $WO

cd ..

# MDOE=(mild severe nonuniform)
# # MDOE=(severe)
# DATANAME=(geo spot)
# CUDA=cuda:0
# # CAM=(gradcam gradcam++ xgradcam)
# CAM=(xgradcam)
# # DEHAZE=(dcp) #(dcp aod ffa)
# DEHAZE=(none)
# NET=fsan
# WO=dcfl # (none, dcfl, darm, dpo)

# cd train/
# for dataname in ${DATANAME[@]}
# do
#     # python classification_lw_woDARM.py --mode clear --data_name $dataname --cls_model resnet --device $CUDA --batchsize1 1
#     for mode in ${MDOE[@]}
#     do
#         for c in ${CAM[@]}
#         do
#             for dh in ${DEHAZE[@]}
#             do
#                 echo $mode $dataname $c $dh $WO
#                 # python classification_lw_woDCFL.py --mode $mode --data_name $dataname --cls_model resnet --device $CUDA --batchsize1 4 --cam $c

#                 # python make_cam_lw_woDCFL.py --mode $mode --data_name $dataname --device $CUDA --cam $c

#                 if [ $NET = "fsan" ]; then
#                     # python segmentation/mainNet_fsan.py --mode $mode --data_name $dataname --device $CUDA --learning_rate 0.001 --cam $c --dehaze $dh --net $NET --wo $WO
#                     python segmentation/predict_fsan.py --mode $mode --data_name $dataname --device $CUDA --cam $c --dehaze $dh --net $NET --wo $WO
#                 elif [ $NET = "refinenet" ]; then
#                     python segmentation/mainNet.py --mode $mode --data_name $dataname --device $CUDA --learning_rate 0.0005 --cam $c --dehaze $dh --net $NET --wo $WO
#                     python segmentation/predict.py --mode $mode --data_name $dataname --device $CUDA --cam $c --net $NET --wo $WO
#                 fi
#             done

#         done
#     done
# done
# cd ..