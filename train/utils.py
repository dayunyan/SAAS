from torchvision import transforms
import argparse
import sys, os

sys.path.append("../")
script_dir = os.path.dirname(os.path.realpath(__file__))

DATA_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Weakly Supervised Semantic Segmentation"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="clear",
        choices=["clear", "mild", "severe", "nonuniform"],
        help="type of haze",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        required=False,
        default="geo",
        choices=["geo", "spot", "love"],
        help="name of datasets",
    )
    parser.add_argument(
        "--cls_model",
        type=str,
        default="resnet",
        help="the model in classification stage",
    )
    parser.add_argument(
        "--batchsize1", type=int, default=16, help="batchsize of classification stage"
    )
    # parser.add_argument("--batchsize2", type=int, default=BATCH_SIZE2, help="batchsize of segmentation stage")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument(
        "--T",
        required=False,
        default=DATA_TRANSFORM,
        help="data transform of classification",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cam", type=str, default="cam")

    return parser.parse_args()


args = get_arguments()

if args.data_name == "geo" or args.data_name == "spot":
    if args.data_name == "geo":
        from configs.config import config
    else:
        from configs.configSPOT import config
else:
    from configs.configLoveDA import config

if args.mode == "clear":
    args.mode = config.CLEAR
elif args.mode == "mild":
    args.mode = config.HAZE
elif args.mode == "severe":
    args.mode = config.DEEPHAZE
else:
    args.mode = config.NONUNIHAZE

args.num_classes = config.num_classes
print("------------utils----------------")
