from pathlib import Path
from PIL import Image

import numpy as np

def load_image(
		img_name, # Name.extension string
		scale_factor=1.0,
		section = "poseterior",
		img_type = "HE",
		img_mode = 'RGB'
	):
	
	data_dir = Path('.')/'data'/section/img_type
	assert data_dir.exists(), f"The data directory {data_dir} does not exist.\nPlease consider making a symbolic link here pointing to your data."
	
	if scale_factor == 1:
		img_path = data_dir/f'{img_name}'
		assert img_path.exists(), f"The image {img_name} does not exist at {data_dir}."
		pil_image = Image.open(img_path).convert(img_mode)
		image = np.asarray(pil_image).copy()
		return image

	if scale_factor != 1:
		img_path = data_dir/f'scale={scale_factor}'/f'{img_name}'
		assert img_path.exists(), f"The image {img_name} does not exist at {data_dir}."
		pil_image = Image.open(img_path).convert(img_mode)
		image = np.asarray(pil_image).copy()
		return image
	
def scale_save_image(
		img_name, # Name.extension string
		scale_factor=1.0,
		section = "poseterior",
		img_type = "HE",
		img_mode = 'RGB'
	):
	
	data_dir = Path('.')/'data'/section/img_type
	assert data_dir.exists(), f"The data directory {data_dir} does not exist.\nPlease consider making a symbolic link here pointing to your data."

	if data_dir.is_symlink():
		data_dir = data_dir.resolve()
		print(f"The provided path is a symlink. Data will be stored at pointed-at path:\n\t{data_dir}")

	data_dir.mkdir(parents=True, exist_ok=True)
