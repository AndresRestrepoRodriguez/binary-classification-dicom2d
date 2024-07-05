import argparse
from utils.data import (
    read_yaml,
    download_public_google_drive_file,
    decompress_file,
    define_val_transformation,
    define_dataloader
)
from data.DICOMDataset import BinaryDICOMDataset
from models.validation_model import validate_model
from utils.schemas import BinaryDataModel
from pathlib import Path
import sys
import os


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def val(opt):

    batch_size, weights, data, save_dir_data, image_size = (
        opt.batch_size,
        opt.weights,
        opt.data,
        opt.save_dir_data,
        opt.imgsz
    )

    data_cfg = read_yaml(data)
    data_model = BinaryDataModel(**data_cfg)

    destination_file = os.path.join(save_dir_data, 'tmp_dataset.' + data_model.extension)

    download_public_google_drive_file(data_model.id_file,
                                      destination_file)
    
    decompress_file(destination_file,
                    os.path.join(save_dir_data, data_model.folder))
    
    dataset_extracted = os.path.join(save_dir_data,
                                     data_model.folder)
    
    transformations_validation = define_val_transformation(image_size)

    validation_dataset = BinaryDICOMDataset(os.path.join(dataset_extracted, 'val'),
                                            transform=transformations_validation)
    
    validation_dataloader = define_dataloader(validation_dataset, batch_size, shuffle=False)

   
    validate_model(
        validation_dataloader,
        weights
        )


def parse_opt(known=False):
    """Parses command-line arguments for YOLOv5 training, validation, and testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")
    parser.add_argument("--data", type=str, default=None, help="dataset.yaml path")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    parser.add_argument("--device", default="cuda", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--save-dir-data", default=ROOT / "data/raw", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")


    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(parameters):
    val(parameters)


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