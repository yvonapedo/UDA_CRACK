from pathlib import Path

data_path_root = Path(r'C:\Users\yvona\Downloads\Curri-AFDA-main\Curri-AFDA-main\data')

def get_data_paths_list(domain='Domain1', split='train', type='image', data_path_root=data_path_root):
    paths_list = list((data_path_root / domain / split / type).glob('*'))
    return paths_list
