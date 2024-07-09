import argparse
from models.DICOMBinaryClassification import BinaryClassificationCNN
import torch
from models.export_model import export_model_pytorch_trace
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
    
    im = torch.zeros(1, 1, image_size, image_size)

    weights_name = os.path.basename(weights)
    export_model_pytorch_trace(model=model,
                               im=im,
                               torchscript_file_path=os.path.splitext(weights_name)[0] + '.torchscript')
    


def parse_opt(known=False):
    """Parses command-line arguments for YOLOv5 training, validation, and testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(parameters):
    export(parameters)


def run(**kwargs):
    """
    Executes YOLOv5 training with given options, overriding with any kwargs provided.

    Example: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
