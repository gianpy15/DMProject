import os

def create_if_does_not_exist(path):
    split_folder = os.path.split(path)
    if '.' in split_folder[1]:
        # path is a file
        path = split_folder[0]
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f'{path} folder created')
