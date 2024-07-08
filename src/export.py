import argparse
from models.DICOMBinaryClassification import BinaryClassificationCNN
import torch
from models.export_model import export_model_pytorch
import os




def export(opt):

    weights, image_size = (
        opt.weights,
        opt.imgsz
    )

    if weights:
        try:
            model = BinaryClassificationCNN()
            model.load_state_dict(torch.load(weights))

        except Exception as e:
            raise Exception(f'Problem loading the Model: {e}')

    else:
        raise Exception(f'Weights path is required')
    

    weights_name = os.path.basename(weights)
    export_model_pytorch(model=model,
                         torchscript_file_path=weights_name + '.torchscript')
    


def parse_opt(known=False):
    """Parses command-line arguments for YOLOv5 training, validation, and testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")

    return parser.parse_known_args()[0] if known else parser.parse_args()
