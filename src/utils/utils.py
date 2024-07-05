import os


def check_folder_existence(directory_path: str):
    if os.listdir(directory_path) == []:
        return False
    else:
        return True