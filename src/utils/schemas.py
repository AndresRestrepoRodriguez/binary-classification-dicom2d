from pydantic import BaseModel
from typing import List


class BinaryDataModel(BaseModel):
    source: str
    id_file: str
    task: str
    classes: List[str]
    extension: str
    folder: str


class BinaryPytorchModel(BaseModel):
    source: str
    id_file: str
    task: str
    classes: List[str]
    extension: str
    framework: str
    file_name: str
    folder: str