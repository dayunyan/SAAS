import os
from easydict import EasyDict as edict
import json
import torch

config = edict()
config.NAME = "geo"

config.weight = 768
config.height = 768
# config.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# config.device = torch.device('cpu')
root = "/home/zjj/xjd/"
dataset_path = os.path.join(root, "datasets/geoeye-1")
psm_path = os.path.join(root, "psm/geoeye-1")
model_path = os.path.join(root, "SAAS/models/save/geo")

haze_mode = ["CLEAR", "HAZE", "DEEPHAZE", "NONUNIHAZE", "TRUE"]
haze_mode2 = ["", "mild", "severe", "nonuniform", "true"]
dehaze_mode = ["", "_dehaze", "_AODdehaze", "_FFAdehaze", "_APFFdehaze"]
model_mode = ["", "_nonconsist", "_woAtt", "_woPerloss", "_recons"]

normal_key = ["name", "model_name", "aux_postive_dir"]
model_key = [
    f"model{m}_path" for m in model_mode
]  # 后续需要改一下所有model_path_nonconsist -> model_nonconsist_path, pcares_woAtt_perloss_path -> model_woAtt_path, pcares_woPerloss_perloss_path -> model_woPerloss_path
train_key = [f"train{d}_dir" for d in dehaze_mode]
val_key = [f"val{d}_dir" for d in dehaze_mode]
test_key = [f"test{d}_dir" for d in dehaze_mode]
aux_key = [f"aux{d}_dir" for d in dehaze_mode]

# 伪显著图key：完整模型 + 各去雾方法 + 各消融模块 (train_perloss_psm ->train_nonconsist_psm, train_pcares_woAtt_perloss_psm -> train_woAtt_psm, train_pcares_woPerloss_psm -> train_woPerloss_psm)
train_psm_key = [f"train{d}_psm" for d in dehaze_mode] + [
    f"train{m}_psm" for m in model_mode[1:]
]
#     "train_perloss_psm",
#     "train_pcares_perloss_psm", 已弃用
#     "train_pcares_woAtt_perloss_psm",
#     "train_pcares_woPerloss_psm",
# ]
val_psm_key = [f"val{d}_psm" for d in dehaze_mode] + [
    f"val{m}_psm" for m in model_mode[1:]
]

test_psm_key = [f"test{d}_psm" for d in dehaze_mode] + [
    f"test{m}_psm" for m in model_mode[1:]
]

aux_psm_key = [f"aux{d}_psm" for d in dehaze_mode] + [
    f"aux{m}_psm" for m in model_mode[1:]
]

full_train_key = [f"full{d}_train" for d in dehaze_mode] + [
    f"full{m}_train" for m in model_mode[1:]
]

full_train_gt_key = [f"full{d}_train_gt" for d in dehaze_mode] + [
    f"full{m}_train_gt" for m in model_mode[1:]
]


keys = [
    model_key,
    train_key,
    val_key,
    test_key,
    aux_key,
    train_psm_key,
    val_psm_key,
    test_psm_key,
    aux_psm_key,
    full_train_key,
    full_train_gt_key,
]
for hm, hm2 in list(zip(haze_mode, haze_mode2)):
    config[hm] = edict()
    config[hm]["name"] = hm.lower()
    config[hm]["model_name"] = f"resnet50_{hm.lower()}" if hm != "CLEAR" else "resnet50"
    config[hm]["aux_postive_dir"] = (
        f"{dataset_path}/"
        + (f"haze/{hm2}/" if hm != "CLEAR" else "")
        + "auxiliary/residential/"
    )
    if hm == "CLEAR":
        config.CLEAR.train_dir = f"{dataset_path}/geo_train/train/"
        config.CLEAR.val_dir = f"{dataset_path}/geo_train/validation/"
        config.CLEAR.test_dir = f"{dataset_path}/geo_test/img/"
        config.CLEAR.aux_dir = f"{dataset_path}/auxiliary/full_aux/"

        config.CLEAR.train_psm = f"{psm_path}/train_psm/"
        config.CLEAR.val_psm = f"{psm_path}/val_psm/"
        config.CLEAR.test_psm = f"{psm_path}/test_psm/"
        config.CLEAR.aux_psm = f"{psm_path}/aux_psm/"
        config.CLEAR.full_train = f"{dataset_path}/full_residential/"
        config.CLEAR.full_train_gt = f"{psm_path}/smap_full/"

        config.CLEAR.model_path = f"{model_path}/model_resnet50_best.pth"
        config.CLEAR.model_nonconsist_path = (
            f"{model_path}/model_resnet50_nonconsis_best.pth"
        )
        config.CLEAR.pcares_path = f"{model_path}/model_pcares_best.pth"
        config.CLEAR.model_woAtt_path = f"{model_path}/model_resnet50_woAtt_best.pth"
        config.CLEAR.model_recons_path = f"{model_path}/model_resnet50_recons_best.pth"
        continue
    # 完整的模型 + 不同的消融模型 (model_resnet50_haze_nonconsis_best -> model_resnet50_nonconsist_haze_best, model_pcares_woAtt_haze_perloss_best -> model_resnet50_woAtt_haze_best, model_pcares_woPerloss_haze_best -> model_resnet50_woPerloss_haze_best)
    model_value = [
        f"{model_path}/model_resnet50{m}_{hm.lower()}_best.pth" for m in model_mode
    ]
    # 原始的数据集（训练、验证、测试、全）+ 去雾后的数据集
    train_value = [f"{dataset_path}/haze/{hm2}/train/"] + [
        f"{dataset_path}/haze/{hm2}/{d.split('_')[-1]}/train/" for d in dehaze_mode[1:]
    ]
    val_value = [f"{dataset_path}/haze/{hm2}/val/"] + [
        f"{dataset_path}/haze/{hm2}/{d.split('_')[-1]}/val/" for d in dehaze_mode[1:]
    ]
    test_value = [f"{dataset_path}/haze/{hm2}/test/"] + [
        f"{dataset_path}/dehaze/{hm2}/{d.split('_')[-1]}_test/" for d in dehaze_mode[1:]
    ]  # AOD_dehaze_test/test/ -> AODdehaze_test/
    aux_value = [f"{dataset_path}/haze/{hm2}/auxiliary/"] + [
        f"{dataset_path}/haze/{hm2}/{d.split('_')[-1]}/aux_resi/"
        for d in dehaze_mode[1:]
    ]
    # 伪显著图生成：完整模型 + 各去雾方法 + 各消融模块 （AOD_dehaze -> AODdehaze, perloss_train_haze_psm -> nonconsist_train_haze_psm, pcares_woAtt_perloss -> woAtt, pcares_woPerloss -> woPerloss）
    train_psm_value = [
        f"{psm_path}/{d.split('_')[-1]}train_{hm.lower()}_psm/" for d in dehaze_mode
    ] + [
        f"{psm_path}/{m.split('_')[-1]}_train_{hm.lower()}_psm/" for m in model_mode[1:]
    ]
    val_psm_value = [
        f"{psm_path}/{d.split('_')[-1]}val_{hm.lower()}_psm/" for d in dehaze_mode
    ] + [f"{psm_path}/{m.split('_')[-1]}_val_{hm.lower()}_psm/" for m in model_mode[1:]]
    test_psm_value = [
        f"{psm_path}/{d.split('_')[-1]}test_{hm.lower()}_psm/" for d in dehaze_mode
    ] + [
        f"{psm_path}/{m.split('_')[-1]}_test_{hm.lower()}_psm/" for m in model_mode[1:]
    ]
    aux_psm_value = [
        f"{psm_path}/{d.split('_')[-1]}aux_{hm.lower()}_psm/" for d in dehaze_mode
    ] + [f"{psm_path}/{m.split('_')[-1]}_aux_{hm.lower()}_psm/" for m in model_mode[1:]]
    # 用于分割模型训练的数据
    full_train_value = (
        [f"{dataset_path}/haze/{hm2}/full_train/"]
        + [
            f"{dataset_path}/dehaze/{hm2}/{d.split('_')[-1]}full_train/"
            for d in dehaze_mode[1:]
        ]
        + [
            f"{dataset_path}/haze/{hm2}/{m.split('_')[-1]}_full_train/"
            for m in model_mode[1:]
        ]
    )
    full_train_gt_value = [
        f"{psm_path}/{d.split('_')[-1]}smap_{hm.lower()}_full/" for d in dehaze_mode
    ] + [
        f"{psm_path}/{m.split('_')[-1]}_smap_{hm.lower()}_full/" for m in model_mode[1:]
    ]

    values = [
        model_value,
        train_value,
        val_value,
        test_value,
        aux_value,
        train_psm_value,
        val_psm_value,
        test_psm_value,
        aux_psm_value,
        full_train_value,
        full_train_gt_value,
    ]
    for key, value in zip(keys, values):
        for k, v in zip(key, value):
            config[hm][k] = v

# check config
with open(os.path.join(root, "SAAS/configs/config_info.txt"), "w") as file:
    for key, value in config.items():
        if isinstance(value, edict):
            for k, v in value.items():
                # 格式化输出键和值到文件
                file.write(f"{k}: {v}\n")

        else:
            file.write(f"{key}: {value}\n")

##################################################
# config.CLEAR = edict()
# config.CLEAR.name = "clear"
# config.CLEAR.train_dir = f"{dataset_path}/geo_train/train/"
# config.CLEAR.val_dir = f"{dataset_path}/geo_train/validation/"
# config.CLEAR.test_dir = f"{dataset_path}/geo_test/img/"
# config.CLEAR.aux_dir = f"{dataset_path}/auxiliary/full_aux/"
# config.CLEAR.aux_postive_dir = f"{dataset_path}/auxiliary/residential/"

# config.CLEAR.train_psm = f"{psm_path}/train_psm/"
# config.CLEAR.val_psm = f"{psm_path}/val_psm/"
# config.CLEAR.test_psm = f"{psm_path}/test_psm/"
# config.CLEAR.aux_psm = f"{psm_path}/aux_psm/"
# config.CLEAR.full_train = f"{dataset_path}/full_residential/"
# config.CLEAR.full_train_gt = f"{psm_path}/smap_full/"

# config.CLEAR.model_name = "resnet50"
# config.CLEAR.model_path = f"{model_path}/model_resnet50_best.pth"
# config.CLEAR.model_path_nonconsist = f"{model_path}/model_resnet50_nonconsis_best.pth"
# config.CLEAR.pcares_path = f"{model_path}/model_pcares_best.pth"
# config.CLEAR.pcares_woAtt_path = f"{model_path}/model_pcares_woAtt_best.pth"
# ##################################################
# config.HAZE = edict()
# config.HAZE.name = "haze"
# config.HAZE.train_dir = f"{dataset_path}/haze/mild/train/"
# config.HAZE.val_dir = f"{dataset_path}/haze/mild/val/"
# config.HAZE.test_dir = f"{dataset_path}/haze/mild/test/"
# config.HAZE.aux_dir = f"{dataset_path}/haze/mild/auxiliary/"
# config.HAZE.aux_postive_dir = f"{dataset_path}/haze/mild/auxiliary/residential/"

# config.HAZE.train_dehaze_dir = f"{dataset_path}/haze/mild/dehaze/train/"
# config.HAZE.val_dehaze_dir = f"{dataset_path}/haze/mild/dehaze/val/"
# config.HAZE.test_dehaze_dir = f"{dataset_path}/dehaze/mild/dehaze_test/test/"
# config.HAZE.aux_dehaze_dir = f"{dataset_path}/haze/mild/dehaze/aux_resi/"

# config.HAZE.train_AODdehaze_dir = f"{dataset_path}/haze/mild/AODdehaze/train/"
# config.HAZE.val_AODdehaze_dir = f"{dataset_path}/haze/mild/AODdehaze/val/"
# config.HAZE.test_AODdehaze_dir = f"{dataset_path}/dehaze/mild/AOD_dehaze_test/test/"
# config.HAZE.aux_AODdehaze_dir = f"{dataset_path}/haze/mild/AODdehaze/aux_resi/"

# config.HAZE.train_FFAdehaze_dir = f"{dataset_path}/haze/mild/FFAdehaze/train/"
# config.HAZE.val_FFAdehaze_dir = f"{dataset_path}/haze/mild/FFAdehaze/val/"
# config.HAZE.test_FFAdehaze_dir = f"{dataset_path}/dehaze/mild/FFA_dehaze_test/test/"
# config.HAZE.aux_FFAdehaze_dir = f"{dataset_path}/haze/mild/FFAdehaze/aux_resi/"

# config.HAZE.train_APFFdehaze_dir = f"{dataset_path}/haze/mild/APFFdehaze/train/"
# config.HAZE.val_APFFdehaze_dir = f"{dataset_path}/haze/mild/APFFdehaze/val/"
# config.HAZE.test_APFFdehaze_dir = f"{dataset_path}/haze/mild/APFF_dehaze_test/test/"
# config.HAZE.aux_APFFdehaze_dir = f"{dataset_path}/haze/mild/APFFdehaze/aux_resi/"

# config.HAZE.train_psm = f"{psm_path}/train_haze_psm/"
# config.HAZE.val_psm = f"{psm_path}/val_haze_psm/"
# config.HAZE.test_psm = f"{psm_path}/test_haze_psm/"
# config.HAZE.aux_psm = f"{psm_path}/aux_haze_psm/"
# config.HAZE.full_train = f"{dataset_path}/haze/mild/full_train/"
# config.HAZE.full_train_gt = f"{psm_path}/smap_haze_full/"

# config.HAZE.train_perloss_psm = f"{psm_path}/perloss_train_haze_psm/"
# config.HAZE.val_perloss_psm = f"{psm_path}/perloss_val_haze_psm/"
# config.HAZE.test_perloss_psm = f"{psm_path}/perloss_test_haze_psm/"
# config.HAZE.aux_perloss_psm = f"{psm_path}/perloss_aux_haze_psm/"
# config.HAZE.full_perloss_train = f"{dataset_path}/haze/mild/perloss_full_train/"
# config.HAZE.full_perloss_train_gt = f"{psm_path}/perloss_smap_haze_full/"

# config.HAZE.train_pcares_perloss_psm = f"{psm_path}/pcares_perloss_train_haze_psm/"
# config.HAZE.val_pcares_perloss_psm = f"{psm_path}/pcares_perloss_val_haze_psm/"
# config.HAZE.test_pcares_perloss_psm = f"{psm_path}/pcares_perloss_test_haze_psm/"
# config.HAZE.aux_pcares_perloss_psm = f"{psm_path}/pcares_perloss_aux_haze_psm/"
# config.HAZE.full_pcares_perloss_train = (
#     f"{dataset_path}/haze/mild/pcares_perloss_full_train/"
# )
# config.HAZE.full_pcares_perloss_train_gt = f"{psm_path}/pcares_perloss_smap_haze_full/"

# # w/o DARM
# config.HAZE.train_pcares_woAtt_perloss_psm = (
#     f"{psm_path}/pcares_woAtt_perloss_train_haze_psm/"
# )
# config.HAZE.val_pcares_woAtt_perloss_psm = (
#     f"{psm_path}/pcares_woAtt_perloss_val_haze_psm/"
# )
# config.HAZE.test_pcares_woAtt_perloss_psm = (
#     f"{psm_path}/pcares_woAtt_perloss_test_haze_psm/"
# )
# config.HAZE.aux_pcares_woAtt_perloss_psm = (
#     f"{psm_path}/pcares_woAtt_perloss_aux_haze_psm/"
# )
# config.HAZE.full_pcares_woAtt_perloss_train = (
#     f"{dataset_path}/haze/mild/pcares_woAtt_perloss_full_train/"
# )
# config.HAZE.full_pcares_woAtt_perloss_train_gt = (
#     f"{psm_path}/pcares_woAtt_perloss_smap_haze_full/"
# )

# # w/o DPO
# config.HAZE.train_pcares_woPerloss_psm = f"{psm_path}/pcares_woPerloss_train_haze_psm/"
# config.HAZE.val_pcares_woPerloss_psm = f"{psm_path}/pcares_woPerloss_val_haze_psm/"
# config.HAZE.test_pcares_woPerloss_psm = f"{psm_path}/pcares_woPerloss_test_haze_psm/"
# config.HAZE.aux_pcares_woPerloss_psm = f"{psm_path}/pcares_woPerloss_aux_haze_psm/"
# config.HAZE.full_pcares_woPerloss_train = (
#     f"{dataset_path}/haze/mild/pcares_woPerloss_full_train/"
# )
# config.HAZE.full_pcares_woPerloss_train_gt = (
#     f"{psm_path}/pcares_woPerloss_smap_haze_full/"
# )

# config.HAZE.train_dehaze_psm = f"{psm_path}/dehazetrain_haze_psm/"
# config.HAZE.val_dehaze_psm = f"{psm_path}/dehazeval_haze_psm/"
# config.HAZE.test_dehaze_psm = f"{psm_path}/dehazetest_haze_psm/"
# config.HAZE.aux_dehaze_psm = f"{psm_path}/dehazeaux_haze_psm/"
# config.HAZE.full_dehaze_train = f"{dataset_path}/dehaze/mild/dehazefull_train/"
# config.HAZE.full_dehaze_train_gt = f"{psm_path}/dehazesmap_haze_full/"

# config.HAZE.train_AODdehaze_psm = f"{psm_path}/AOD_dehazetrain_haze_psm/"
# config.HAZE.val_AODdehaze_psm = f"{psm_path}/AOD_dehazeval_haze_psm/"
# config.HAZE.test_AODdehaze_psm = f"{psm_path}/AOD_dehazetest_haze_psm/"
# config.HAZE.aux_AODdehaze_psm = f"{psm_path}/AOD_dehazeaux_haze_psm/"
# config.HAZE.full_AODdehaze_train = f"{dataset_path}/dehaze/mild/AOD_dehazefull_train/"
# config.HAZE.full_AODdehaze_train_gt = f"{psm_path}/AOD_dehazesmap_haze_full/"

# config.HAZE.train_FFAdehaze_psm = f"{psm_path}/FFA_dehazetrain_haze_psm/"
# config.HAZE.val_FFAdehaze_psm = f"{psm_path}/FFA_dehazeval_haze_psm/"
# config.HAZE.test_FFAdehaze_psm = f"{psm_path}/FFA_dehazetest_haze_psm/"
# config.HAZE.aux_FFAdehaze_psm = f"{psm_path}/FFA_dehazeaux_haze_psm/"
# config.HAZE.full_FFAdehaze_train = f"{dataset_path}/dehaze/mild/FFA_dehazefull_train/"
# config.HAZE.full_FFAdehaze_train_gt = f"{psm_path}/FFA_dehazesmap_haze_full/"

# config.HAZE.train_APFFdehaze_psm = f"{psm_path}/APFF_dehazetrain_haze_psm/"
# config.HAZE.val_APFFdehaze_psm = f"{psm_path}/APFF_dehazeval_haze_psm/"
# config.HAZE.test_APFFdehaze_psm = f"{psm_path}/APFF_dehazetest_haze_psm/"
# config.HAZE.aux_APFFdehaze_psm = f"{psm_path}/APFF_dehazeaux_haze_psm/"
# config.HAZE.full_APFFdehaze_train = f"{dataset_path}/haze/mild/APFF_dehazefull_train/"
# config.HAZE.full_APFFdehaze_train_gt = f"{psm_path}/APFF_dehazesmap_haze_full/"

# config.HAZE.model_name = "resnet50_haze"
# config.HAZE.model_path = f"{model_path}/model_resnet50_haze_best.pth"
# config.HAZE.model_path_nonconsist = (
#     f"{model_path}/model_resnet50_haze_nonconsis_best.pth"
# )
# config.HAZE.model_perloss_path = f"{model_path}/model_resnet50_haze_perloss_best.pth"
# config.HAZE.pcares_perloss_path = f"{model_path}/model_pcares_haze_perloss_best.pth"
# config.HAZE.pcares_woAtt_perloss_path = (
#     f"{model_path}/model_pcares_woAtt_haze_perloss_best.pth"
# )
# config.HAZE.pcares_woPerloss_path = f"{model_path}/model_pcares_woPerloss_haze_best.pth"
# ##################################################
# config.DEEPHAZE = edict()
# config.DEEPHAZE.name = "deephaze"
# config.DEEPHAZE.train_dir = f"{dataset_path}/haze/severe/train/"
# config.DEEPHAZE.val_dir = f"{dataset_path}/haze/severe/val/"
# config.DEEPHAZE.test_dir = f"{dataset_path}/haze/severe/test/"
# config.DEEPHAZE.aux_dir = f"{dataset_path}/haze/severe/auxiliary/full/"
# config.DEEPHAZE.aux_postive_dir = f"{dataset_path}/haze/severe/auxiliary/residential/"

# config.DEEPHAZE.train_dehaze_dir = f"{dataset_path}/haze/severe/dehaze/train/"
# config.DEEPHAZE.val_dehaze_dir = f"{dataset_path}/haze/severe/dehaze/val/"
# config.DEEPHAZE.test_dehaze_dir = f"{dataset_path}/dehaze/severe/dehaze_test/test/"
# config.DEEPHAZE.aux_dehaze_dir = f"{dataset_path}/haze/severe/dehaze/aux_resi/"

# config.DEEPHAZE.train_AODdehaze_dir = f"{dataset_path}/haze/severe/AODdehaze/train/"
# config.DEEPHAZE.val_AODdehaze_dir = f"{dataset_path}/haze/severe/AODdehaze/val/"
# config.DEEPHAZE.test_AODdehaze_dir = (
#     f"{dataset_path}/dehaze/severe/AOD_dehaze_test/test/"
# )
# config.DEEPHAZE.aux_AODdehaze_dir = f"{dataset_path}/haze/severe/AODdehaze/aux_resi/"

# config.DEEPHAZE.train_FFAdehaze_dir = f"{dataset_path}/haze/severe/FFAdehaze/train/"
# config.DEEPHAZE.val_FFAdehaze_dir = f"{dataset_path}/haze/severe/FFAdehaze/val/"
# config.DEEPHAZE.test_FFAdehaze_dir = (
#     f"{dataset_path}/dehaze/severe/FFA_dehaze_test/test/"
# )
# config.DEEPHAZE.aux_FFAdehaze_dir = f"{dataset_path}/haze/severe/FFAdehaze/aux_resi/"

# config.DEEPHAZE.train_APFFdehaze_dir = f"{dataset_path}/haze/severe/APFFdehaze/train/"
# config.DEEPHAZE.val_APFFdehaze_dir = f"{dataset_path}/haze/severe/APFFdehaze/val/"
# config.DEEPHAZE.test_APFFdehaze_dir = (
#     f"{dataset_path}/haze/severe/APFF_dehaze_test/test/"
# )
# config.DEEPHAZE.aux_APFFdehaze_dir = f"{dataset_path}/haze/severe/APFFdehaze/aux_resi/"

# config.DEEPHAZE.train_psm = f"{psm_path}/train_deephaze_psm/"
# config.DEEPHAZE.val_psm = f"{psm_path}/val_deephaze_psm/"
# config.DEEPHAZE.test_psm = f"{psm_path}/test_deephaze_psm/"
# config.DEEPHAZE.aux_psm = f"{psm_path}/aux_deephaze_psm/"
# config.DEEPHAZE.full_train = f"{dataset_path}/haze/severe/full_train/"
# config.DEEPHAZE.full_train_gt = f"{psm_path}/smap_deephaze_full/"

# config.DEEPHAZE.train_perloss_psm = f"{psm_path}/perloss_train_deephaze_psm/"
# config.DEEPHAZE.val_perloss_psm = f"{psm_path}/perloss_val_deephaze_psm/"
# config.DEEPHAZE.test_perloss_psm = f"{psm_path}/perloss_test_deephaze_psm/"
# config.DEEPHAZE.aux_perloss_psm = f"{psm_path}/perloss_aux_deephaze_psm/"
# config.DEEPHAZE.full_perloss_train = f"{dataset_path}/haze/severe/perloss_full_train/"
# config.DEEPHAZE.full_perloss_train_gt = f"{psm_path}/perloss_smap_deephaze_full/"

# config.DEEPHAZE.train_pcares_perloss_psm = (
#     f"{psm_path}/pcares_perloss_train_deephaze_psm/"
# )
# config.DEEPHAZE.val_pcares_perloss_psm = f"{psm_path}/pcares_perloss_val_deephaze_psm/"
# config.DEEPHAZE.test_pcares_perloss_psm = (
#     f"{psm_path}/pcares_perloss_test_deephaze_psm/"
# )
# config.DEEPHAZE.aux_pcares_perloss_psm = f"{psm_path}/pcares_perloss_aux_deephaze_psm/"
# config.DEEPHAZE.full_pcares_perloss_train = (
#     f"{dataset_path}/haze/severe/pcares_perloss_full_train/"
# )
# config.DEEPHAZE.full_pcares_perloss_train_gt = (
#     f"{psm_path}/pcares_perloss_smap_deephaze_full/"
# )

# config.DEEPHAZE.train_pcares_woAtt_perloss_psm = (
#     f"{psm_path}/pcares_woAtt_perloss_train_deephaze_psm/"
# )
# config.DEEPHAZE.val_pcares_woAtt_perloss_psm = (
#     f"{psm_path}/pcares_woAtt_perloss_val_deephaze_psm/"
# )
# config.DEEPHAZE.test_pcares_woAtt_perloss_psm = (
#     f"{psm_path}/pcares_woAtt_perloss_test_deephaze_psm/"
# )
# config.DEEPHAZE.aux_pcares_woAtt_perloss_psm = (
#     f"{psm_path}/pcares_woAtt_perloss_aux_deephaze_psm/"
# )
# config.DEEPHAZE.full_pcares_woAtt_perloss_train = (
#     f"{dataset_path}/haze/severe/pcares_woAtt_perloss_full_train/"
# )
# config.DEEPHAZE.full_pcares_woAtt_perloss_train_gt = (
#     f"{psm_path}/pcares_woAtt_perloss_smap_deephaze_full/"
# )

# config.DEEPHAZE.train_pcares_woPerloss_psm = (
#     f"{psm_path}/pcares_woPerloss_train_deephaze_psm/"
# )
# config.DEEPHAZE.val_pcares_woPerloss_psm = (
#     f"{psm_path}/pcares_woPerloss_val_deephaze_psm/"
# )
# config.DEEPHAZE.test_pcares_woPerloss_psm = (
#     f"{psm_path}/pcares_woPerloss_test_deephaze_psm/"
# )
# config.DEEPHAZE.aux_pcares_woPerloss_psm = (
#     f"{psm_path}/pcares_woPerloss_aux_deephaze_psm/"
# )
# config.DEEPHAZE.full_pcares_woPerloss_train = (
#     f"{dataset_path}/haze/severe/pcares_woPerloss_full_train/"
# )
# config.DEEPHAZE.full_pcares_woPerloss_train_gt = (
#     f"{psm_path}/pcares_woPerloss_smap_deephaze_full/"
# )

# config.DEEPHAZE.train_dehaze_psm = f"{psm_path}/dehazetrain_deephaze_psm/"
# config.DEEPHAZE.val_dehaze_psm = f"{psm_path}/dehazeval_deephaze_psm/"
# config.DEEPHAZE.test_dehaze_psm = f"{psm_path}/dehazetest_deephaze_psm/"
# config.DEEPHAZE.aux_dehaze_psm = f"{psm_path}/dehazeaux_deephaze_psm/"
# config.DEEPHAZE.full_dehaze_train = f"{dataset_path}/dehaze/severe/dehazefull_train/"
# config.DEEPHAZE.full_dehaze_train_gt = f"{psm_path}/dehazesmap_deephaze_full/"

# config.DEEPHAZE.train_AODdehaze_psm = f"{psm_path}/AOD_dehazetrain_deephaze_psm/"
# config.DEEPHAZE.val_AODdehaze_psm = f"{psm_path}/AOD_dehazeval_deephaze_psm/"
# config.DEEPHAZE.test_AODdehaze_psm = f"{psm_path}/AOD_dehazetest_deephaze_psm/"
# config.DEEPHAZE.aux_AODdehaze_psm = f"{psm_path}/AOD_dehazeaux_deephaze_psm/"
# config.DEEPHAZE.full_AODdehaze_train = (
#     f"{dataset_path}/dehaze/severe/AOD_dehazefull_train/"
# )
# config.DEEPHAZE.full_AODdehaze_train_gt = f"{psm_path}/AOD_dehazesmap_deephaze_full/"

# config.DEEPHAZE.train_FFAdehaze_psm = f"{psm_path}/FFA_dehazetrain_deephaze_psm/"
# config.DEEPHAZE.val_FFAdehaze_psm = f"{psm_path}/FFA_dehazeval_deephaze_psm/"
# config.DEEPHAZE.test_FFAdehaze_psm = f"{psm_path}/FFA_dehazetest_deephaze_psm/"
# config.DEEPHAZE.aux_FFAdehaze_psm = f"{psm_path}/FFA_dehazeaux_deephaze_psm/"
# config.DEEPHAZE.full_FFAdehaze_train = (
#     f"{dataset_path}/dehaze/severe/FFA_dehazefull_train/"
# )
# config.DEEPHAZE.full_FFAdehaze_train_gt = f"{psm_path}/FFA_dehazesmap_deephaze_full/"

# config.DEEPHAZE.train_APFFdehaze_psm = f"{psm_path}/APFF_dehazetrain_deephaze_psm/"
# config.DEEPHAZE.val_APFFdehaze_psm = f"{psm_path}/APFF_dehazeval_deephaze_psm/"
# config.DEEPHAZE.test_APFFdehaze_psm = f"{psm_path}/APFF_dehazetest_deephaze_psm/"
# config.DEEPHAZE.aux_APFFdehaze_psm = f"{psm_path}/APFF_dehazeaux_deephaze_psm/"
# config.DEEPHAZE.full_APFFdehaze_train = (
#     f"{dataset_path}/haze/severe/APFF_dehazefull_train/"
# )
# config.DEEPHAZE.full_APFFdehaze_train_gt = f"{psm_path}/APFF_dehazesmap_deephaze_full/"

# config.DEEPHAZE.model_name = "resnet50_deephaze"
# config.DEEPHAZE.model_path = f"{model_path}/model_resnet50_deephaze_best.pth"
# config.DEEPHAZE.model_path_nonconsist = (
#     f"{model_path}/model_resnet50_deephaze_nonconsis_best.pth"
# )
# config.DEEPHAZE.model_perloss_path = (
#     f"{model_path}/model_resnet50_deephaze_perloss_best.pth"
# )
# config.DEEPHAZE.pcares_perloss_path = (
#     f"{model_path}/model_pcares_deephaze_perloss_best.pth"
# )
# config.DEEPHAZE.pcares_woAtt_perloss_path = (
#     f"{model_path}/model_pcares_woAtt_deephaze_perloss_best.pth"
# )
# config.DEEPHAZE.pcares_woPerloss_path = (
#     f"{model_path}/model_pcares_woPerloss_deephaze_best.pth"
# )
# ##################################################
# config.NONUNIHAZE = edict()
# config.NONUNIHAZE.name = "nonunihaze"
# config.NONUNIHAZE.train_dir = f"{dataset_path}/haze/nonuniform/train/"
# config.NONUNIHAZE.val_dir = f"{dataset_path}/haze/nonuniform/val/"
# config.NONUNIHAZE.test_dir = f"{dataset_path}/haze/nonuniform/test/"
# config.NONUNIHAZE.aux_dir = f"{dataset_path}/haze/nonuniform/auxiliary/full/"
# config.NONUNIHAZE.aux_postive_dir = (
#     f"{dataset_path}/haze/nonuniform/auxiliary/residential/"
# )

# config.NONUNIHAZE.train_dehaze_dir = f"{dataset_path}/haze/nonuniform/dehaze/train/"
# config.NONUNIHAZE.val_dehaze_dir = f"{dataset_path}/haze/nonuniform/dehaze/val/"
# config.NONUNIHAZE.test_dehaze_dir = (
#     f"{dataset_path}/dehaze/nonuniform/dehaze_test/test/"
# )
# config.NONUNIHAZE.aux_dehaze_dir = f"{dataset_path}/haze/nonuniform/dehaze/aux_resi/"

# config.NONUNIHAZE.train_AODdehaze_dir = (
#     f"{dataset_path}/haze/nonuniform/AODdehaze/train/"
# )
# config.NONUNIHAZE.val_AODdehaze_dir = f"{dataset_path}/haze/nonuniform/AODdehaze/val/"
# config.NONUNIHAZE.test_AODdehaze_dir = (
#     f"{dataset_path}/dehaze/nonuniform/AOD_dehaze_test/test/"
# )
# config.NONUNIHAZE.aux_AODdehaze_dir = (
#     f"{dataset_path}/haze/nonuniform/AODdehaze/aux_resi/"
# )

# config.NONUNIHAZE.train_FFAdehaze_dir = (
#     f"{dataset_path}/haze/nonuniform/FFAdehaze/train/"
# )
# config.NONUNIHAZE.val_FFAdehaze_dir = f"{dataset_path}/haze/nonuniform/FFAdehaze/val/"
# config.NONUNIHAZE.test_FFAdehaze_dir = (
#     f"{dataset_path}/dehaze/nonuniform/FFA_dehaze_test/test/"
# )
# config.NONUNIHAZE.aux_FFAdehaze_dir = (
#     f"{dataset_path}/haze/nonuniform/FFAdehaze/aux_resi/"
# )

# config.NONUNIHAZE.train_APFFdehaze_dir = (
#     f"{dataset_path}/haze/nonuniform/APFFdehaze/train/"
# )
# config.NONUNIHAZE.val_APFFdehaze_dir = f"{dataset_path}/haze/nonuniform/APFFdehaze/val/"
# config.NONUNIHAZE.test_APFFdehaze_dir = (
#     f"{dataset_path}/haze/nonuniform/APFF_dehaze_test/test/"
# )
# config.NONUNIHAZE.aux_APFFdehaze_dir = (
#     f"{dataset_path}/haze/nonuniform/APFFdehaze/aux_resi/"
# )

# config.NONUNIHAZE.train_psm = f"{psm_path}/train_nonunihaze_psm/"
# config.NONUNIHAZE.val_psm = f"{psm_path}/val_nonunihaze_psm/"
# config.NONUNIHAZE.test_psm = f"{psm_path}/test_nonunihaze_psm/"
# config.NONUNIHAZE.aux_psm = f"{psm_path}/aux_nonunihaze_psm/"
# config.NONUNIHAZE.full_train = f"{dataset_path}/haze/nonuniform/full_train/"
# config.NONUNIHAZE.full_train_gt = f"{psm_path}/smap_nonunihaze_full/"

# config.NONUNIHAZE.train_perloss_psm = f"{psm_path}/perloss_train_nonunihaze_psm/"
# config.NONUNIHAZE.val_perloss_psm = f"{psm_path}/perloss_val_nonunihaze_psm/"
# config.NONUNIHAZE.test_perloss_psm = f"{psm_path}/perloss_test_nonunihaze_psm/"
# config.NONUNIHAZE.aux_perloss_psm = f"{psm_path}/perloss_aux_nonunihaze_psm/"
# config.NONUNIHAZE.full_perloss_train = (
#     f"{dataset_path}/haze/nonuniform/perloss_full_train/"
# )
# config.NONUNIHAZE.full_perloss_train_gt = f"{psm_path}/perloss_smap_nonunihaze_full/"

# config.NONUNIHAZE.train_pcares_perloss_psm = (
#     f"{psm_path}/pcares_perloss_train_nonunihaze_psm/"
# )
# config.NONUNIHAZE.val_pcares_perloss_psm = (
#     f"{psm_path}/pcares_perloss_val_nonunihaze_psm/"
# )
# config.NONUNIHAZE.test_pcares_perloss_psm = (
#     f"{psm_path}/pcares_perloss_test_nonunihaze_psm/"
# )
# config.NONUNIHAZE.aux_pcares_perloss_psm = (
#     f"{psm_path}/pcares_perloss_aux_nonunihaze_psm/"
# )
# config.NONUNIHAZE.full_pcares_perloss_train = (
#     f"{dataset_path}/haze/nonuniform/pcares_perloss_full_train/"
# )
# config.NONUNIHAZE.full_pcares_perloss_train_gt = (
#     f"{psm_path}/pcares_perloss_smap_nonunihaze_full/"
# )

# config.NONUNIHAZE.train_pcares_woAtt_perloss_psm = (
#     f"{psm_path}/pcares_woAtt_perloss_train_nonunihaze_psm/"
# )
# config.NONUNIHAZE.val_pcares_woAtt_perloss_psm = (
#     f"{psm_path}/pcares_woAtt_perloss_val_nonunihaze_psm/"
# )
# config.NONUNIHAZE.test_pcares_woAtt_perloss_psm = (
#     f"{psm_path}/pcares_woAtt_perloss_test_nonunihaze_psm/"
# )
# config.NONUNIHAZE.aux_pcares_woAtt_perloss_psm = (
#     f"{psm_path}/pcares_woAtt_perloss_aux_nonunihaze_psm/"
# )
# config.NONUNIHAZE.full_pcares_woAtt_perloss_train = (
#     f"{dataset_path}/haze/nonuniform/pcares_woAtt_perloss_full_train/"
# )
# config.NONUNIHAZE.full_pcares_woAtt_perloss_train_gt = (
#     f"{psm_path}/pcares_woAtt_perloss_smap_nonunihaze_full/"
# )

# config.NONUNIHAZE.train_pcares_woPerloss_psm = (
#     f"{psm_path}/pcares_woPerloss_train_nonunihaze_psm/"
# )
# config.NONUNIHAZE.val_pcares_woPerloss_psm = (
#     f"{psm_path}/pcares_woPerloss_val_nonunihaze_psm/"
# )
# config.NONUNIHAZE.test_pcares_woPerloss_psm = (
#     f"{psm_path}/pcares_woPerloss_test_nonunihaze_psm/"
# )
# config.NONUNIHAZE.aux_pcares_woPerloss_psm = (
#     f"{psm_path}/pcares_woPerloss_aux_nonunihaze_psm/"
# )
# config.NONUNIHAZE.full_pcares_woPerloss_train = (
#     f"{dataset_path}/haze/nonuniform/pcares_woPerloss_full_train/"
# )
# config.NONUNIHAZE.full_pcares_woPerloss_train_gt = (
#     f"{psm_path}/pcares_woPerloss_smap_nonunihaze_full/"
# )

# config.NONUNIHAZE.train_dehaze_psm = f"{psm_path}/dehazetrain_nonunihaze_psm/"
# config.NONUNIHAZE.val_dehaze_psm = f"{psm_path}/dehazeval_nonunihaze_psm/"
# config.NONUNIHAZE.test_dehaze_psm = f"{psm_path}/dehazetest_nonunihaze_psm/"
# config.NONUNIHAZE.aux_dehaze_psm = f"{psm_path}/dehazeaux_nonunihaze_psm/"
# config.NONUNIHAZE.full_dehaze_train = (
#     f"{dataset_path}/dehaze/nonuniform/dehazefull_train/"
# )
# config.NONUNIHAZE.full_dehaze_train_gt = f"{psm_path}/dehazesmap_nonunihaze_full/"

# config.NONUNIHAZE.train_AODdehaze_psm = f"{psm_path}/AOD_dehazetrain_nonunihaze_psm/"
# config.NONUNIHAZE.val_AODdehaze_psm = f"{psm_path}/AOD_dehazeval_nonunihaze_psm/"
# config.NONUNIHAZE.test_AODdehaze_psm = f"{psm_path}/AOD_dehazetest_nonunihaze_psm/"
# config.NONUNIHAZE.aux_AODdehaze_psm = f"{psm_path}/AOD_dehazeaux_nonunihaze_psm/"
# config.NONUNIHAZE.full_AODdehaze_train = (
#     f"{dataset_path}/dehaze/nonuniform/AOD_dehazefull_train/"
# )
# config.NONUNIHAZE.full_AODdehaze_train_gt = (
#     f"{psm_path}/AOD_dehazesmap_nonunihaze_full/"
# )

# config.NONUNIHAZE.train_FFAdehaze_psm = f"{psm_path}/FFA_dehazetrain_nonunihaze_psm/"
# config.NONUNIHAZE.val_FFAdehaze_psm = f"{psm_path}/FFA_dehazeval_nonunihaze_psm/"
# config.NONUNIHAZE.test_FFAdehaze_psm = f"{psm_path}/FFA_dehazetest_nonunihaze_psm/"
# config.NONUNIHAZE.aux_FFAdehaze_psm = f"{psm_path}/FFA_dehazeaux_nonunihaze_psm/"
# config.NONUNIHAZE.full_FFAdehaze_train = (
#     f"{dataset_path}/dehaze/nonuniform/FFA_dehazefull_train/"
# )
# config.NONUNIHAZE.full_FFAdehaze_train_gt = (
#     f"{psm_path}/FFA_dehazesmap_nonunihaze_full/"
# )

# config.NONUNIHAZE.train_APFFdehaze_psm = f"{psm_path}/APFF_dehazetrain_nonunihaze_psm/"
# config.NONUNIHAZE.val_APFFdehaze_psm = f"{psm_path}/APFF_dehazeval_nonunihaze_psm/"
# config.NONUNIHAZE.test_APFFdehaze_psm = f"{psm_path}/APFF_dehazetest_nonunihaze_psm/"
# config.NONUNIHAZE.aux_APFFdehaze_psm = f"{psm_path}/APFF_dehazeaux_nonunihaze_psm/"
# config.NONUNIHAZE.full_APFFdehaze_train = (
#     f"{dataset_path}/haze/nonuniform/APFF_dehazefull_train/"
# )
# config.NONUNIHAZE.full_APFFdehaze_train_gt = (
#     f"{psm_path}/APFF_dehazesmap_nonunihaze_full/"
# )

# config.NONUNIHAZE.model_name = "resnet50_nonunihaze"
# config.NONUNIHAZE.model_path = f"{model_path}/model_resnet50_nonunihaze_best.pth"
# config.NONUNIHAZE.model_path_nonconsist = (
#     f"{model_path}/model_resnet50_nonunihaze_nonconsis_best.pth"
# )
# config.NONUNIHAZE.model_perloss_path = (
#     f"{model_path}/model_resnet50_nonunihaze_perloss_best.pth"
# )
# config.NONUNIHAZE.pcares_perloss_path = (
#     f"{model_path}/model_pcares_nonunihaze_perloss_best.pth"
# )
# config.NONUNIHAZE.pcares_woAtt_perloss_path = (
#     f"{model_path}/model_pcares_woAtt_nonunihaze_perloss_best.pth"
# )
# config.NONUNIHAZE.pcares_woPerloss_path = (
#     f"{model_path}/model_pcares_woPerloss_nonunihaze_best.pth"
# )
# ##################################################
config.logfile_dir = "log/"
config.model_file = "/home/zjj/xjd/SAAS/models/save/"

config.batch_size = 32
config.num_epoch = 25
config.num_workers = 0
config.num_classes = 2

# config.model_name = 'resnet50'
# config.model_path = '/home/zjj/xjd/SAAS/models/save/model_resnet50_best.pth'

# config.model_haze_name = 'resnet50_haze'
# config.model_haze_path = '/home/zjj/xjd/SAAS/models/save/model_resnet50_haze_xgradcam_best.pth'
# config.model_nonconsis_path = '/home/zjj/xjd/SAAS/models/save/model_resnet50_nonconsis_best.pth'

## !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:06:25 2020

@author: user
"""
