MDOE=(nonuniform) # (mild severe nonuniform)
DATANAME=(geo)
CUDA=cuda:0
# CAM=(gradcam gradcam++ xgradcam)
CAM=(xgradcam)
DEHAZE=(dcp aod ffa)
# DEHAZE=(none)
NET=fsan
# WO=(none dcfl darm dpo)
WO=(none)

for dataname in ${DATANAME[@]}
do
    for mode in ${MDOE[@]}
    do
        for c in ${CAM[@]}
        do
            for dh in ${DEHAZE[@]}
            do
                for w in ${WO[@]}
                do
                    echo $mode $dataname $c $dh $w
                    if [ $NET = "fsan" ]; then
                        python visual.py --mode $mode --data_name $dataname --device $CUDA --cam $c --dehaze $dh --net $NET --wo $w
                    elif [ $NET = "refinenet" ]; then
                        python visual.py --mode $mode --data_name $dataname --device $CUDA --cam $c --net $NET --wo $w
                    fi
                done
            done

        done
    done
done
