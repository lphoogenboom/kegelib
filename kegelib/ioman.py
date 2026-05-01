from pathlib import Path
from PIL import Image

import numpy as np
import geopandas as gpd

from skimage.transform import rescale
from rasterio.features import rasterize

from kegelib import qupath as klq

def load_image(
		img_name, # Name.extension string
		scale_factor=1.0,
		section = "posterior",
		img_type = "HE",
		img_mode = 'RGB'
	):
	
	data_dir = Path('.')/'data'/section/img_type
	assert data_dir.exists(), f"The data directory {data_dir} does not exist.\nPlease consider making a symbolic link here pointing to your data."
	
	if scale_factor == 1:
		img_path = data_dir/f'{img_name}_HE.ome.tiff'
		assert img_path.exists(), f"The image {img_name} does not exist at {data_dir}."
		pil_image = Image.open(img_path).convert(img_mode)
		image = np.asarray(pil_image).copy()
		return image

	elif scale_factor != 1:
		img_path = data_dir/f'scale={scale_factor:.2f}'/f'{img_name}_scale={scale_factor:.2f}_HE.ome.tiff'
		assert img_path.exists(), f"The image {img_name} does not exist at {data_dir}."
		pil_image = Image.open(img_path).convert(img_mode)
		image = np.asarray(pil_image).copy()
		return image
	
	else:
		return None
	
def load_rasterize_annotation(
		annotation_name,
		scale_factor = 1.0,
		section = "posterior",
		version = 'v2'
	):

	data_dir = Path('.')/'data'/section/'annotations'/version
	assert data_dir.exists(), f"The data directory {data_dir} does not exist.\nPlease consider making a symbolic link here pointing to your data."

	data_dir_scaled = data_dir/f'scale={scale_factor:.2f}'
	data_dir_scaled.mkdir(parents=False, exist_ok=True)

	if scale_factor == 1:
		if not (data_dir_scaled/f"{annotation_name}_scale={scale_factor:.2f}.npy").exists():
			print('Looking for annotation at original scale')
			print(f"Rasterized target file {annotation_name} not found.")
			print(f"Will now rasterize and save before returning")
			
			data_dir = data_dir.resolve()
			image = load_image(annotation_name)
			H,W,_ = image.shape	#type: ignore
			del image

			df_annotation = gpd.read_file(data_dir/f"{annotation_name}.geojson")
			polygon = find_gdf_geometry_name(df_annotation, "retina")

			rasterized_geometry = rasterize(
				[(polygon, 1)], # pixels intersecting with geomtry become 1-valued in matrix            
				out_shape=(H,W), # 2D array as slice of tensor
				# transform=transform,
				fill=0, # pixels outside geometry are 0-valued            
				dtype=np.uint8, # this is the ome-tiff encoding
				all_touched=False # include pixels that partially intersect with geometry?         
			)
			np.save(data_dir_scaled/f'{annotation_name}_scale={scale_factor}.npy', rasterized_geometry)
			print(f'File was saved at {data_dir_scaled}/{annotation_name}_scale={scale_factor}.npy')
			print('Returning rasterized annotation')
			return rasterized_geometry
	
	elif scale_factor != 1:
		print(f"Looking for annotation at scale={scale_factor}")

		if not (data_dir_scaled/f"{annotation_name}_scale={scale_factor:.2f}.npy").exists():
			print(f"Target file {annotation_name} not found.")
			print(f"Will now create file at {data_dir_scaled}...")

			data_dir = data_dir.resolve()
			image = load_image(annotation_name)
			H,W,_ = image.shape	#type: ignore
			del image

			df_annotation = gpd.read_file(data_dir/f"{annotation_name}.geojson")
			polygon = find_gdf_geometry_name(df_annotation, "retina")

			rasterized_geometry = rasterize(
				[(polygon, 1)], # pixels intersecting with geomtry become 1-valued in matrix            
				out_shape=(H,W), # 2D array as slice of tensor
				# transform=transform,
				fill=0, # pixels outside geometry are 0-valued            
				dtype=np.uint8, # this is the ome-tiff encoding
				all_touched=False # include pixels that partially intersect with geometry?         
			)
			print("Annotation was rasterized. Will now downscale...")

			rasterized_geometry = rescale(
				rasterized_geometry,
				scale=scale_factor,
				# mode='constant',
				# cval=0,
				preserve_range=True,
				anti_aliasing=True,
				anti_aliasing_sigma=0.1,
			)
			np.save(data_dir_scaled/f'{annotation_name}_scale={scale_factor}.npy', rasterized_geometry)
			print(f'File was saved at {data_dir_scaled}/{annotation_name}_scale={scale_factor}.npy')
			print('Returning downscaled rasterized annotation')
			return rasterized_geometry
		else:
			return np.load(data_dir_scaled/f"{annotation_name}.npy")
	else:
		return None
		
	
def scale_save_image(
		img_name, # Name without extensions
		scale_factor=1.0,
		section = "posterior",
		img_type = "HE",
	):


	assert scale_factor != 1, "Cannot rescale an image at factor 1.0"

	data_dir = Path('.').resolve()/'data'/section/img_type

	assert data_dir.exists(), f"The data directory {data_dir} does not exist.\nPlease consider making a symbolic link here pointing to your data."

	if data_dir.is_symlink():
		data_dir = data_dir.resolve()
		print(f"\nThe provided path is a symlink. Data will be stored at pointed-at path:\n\t{data_dir}\n")

	scaled_output_dir = data_dir/f'scale={scale_factor:.2f}'

	scaled_output_dir.mkdir(parents=False, exist_ok=True)

	image = load_image(
		img_name=img_name,
		scale_factor = 1.0
		)

	image = rescale(
		image,
		scale=scale_factor,
		# mode='constant',
		# cval=0,
		preserve_range=True,
		anti_aliasing=True,
		anti_aliasing_sigma=0.1,
		channel_axis=-1
	)

	rescale_PIL = Image.fromarray(image.astype(np.uint8), mode='RGB')
	rescale_PIL.save(scaled_output_dir/f'{img_name}_scale={scale_factor:.2f}_HE.ome.tiff')

def find_gdf_geometry_name(geodataframe, annotation_name:str):
	tissue_index = None # row index of desired tissue annotation
	for row_index in range(len(geodataframe)):
		if geodataframe['classification'][row_index] != None:
			geometry_name = eval(geodataframe["classification"][row_index])["name"].lower()
			if geometry_name == "retina":
				tissue_index = row_index

	if type(tissue_index)==int:
		geometry = geodataframe['geometry'].loc[tissue_index]
		return geometry
	elif type(tissue_index)==None:
		print(f'No geometry was found corresponding to name: {annotation_name}')
	else:
		print(f'Unexpected row_index type: {type(row_index)}')