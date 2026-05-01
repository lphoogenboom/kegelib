### Qupath has some quirks when it comes to exporting and importing data
### The focus of this module is to provide some tools to simplify handling data interacting with qupath

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from rasterio import features

import geopandas as gpd

import shapely

from skimage.measure import regionprops_table

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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

def classify_segments_kmeans(clusters, image, labels):

	props = regionprops_table(
		labels, 
		intensity_image=image, # standardized_image
		properties=['label', 'mean_intensity']
	)

	labels_features =  np.array([props['mean_intensity-0'], props['mean_intensity-1'], props['mean_intensity-2']]).T
	labels_ids = props['label']

	_,_,C = image.shape
	initial_centers = np.random.rand(clusters,C)

	kmeans = KMeans(n_clusters=clusters, random_state=42)
	cluster_assignments = kmeans.fit_predict(labels_features)

	label_to_cluster = dict(zip(labels_ids, cluster_assignments))
	classified_map = np.zeros_like(labels)

	unique_labels = np.array(list(label_to_cluster.keys()))

	mapped_clusters = np.array([label_to_cluster[l] for l in unique_labels])

	lookup = np.zeros(labels.max() + 1, dtype=int)

	# lookup = np.arange(labels.max() + 1, dtype=int)
	lookup[unique_labels] = mapped_clusters
	classified_map = 1+lookup[labels]

	# # Get all unique labels present in the image
	# all_labels = np.unique(labels)
	# lookup = np.zeros(all_labels.max() + 1, dtype=int)

	# # Assign clusters where we have data, keep original where we don't
	# for label_id in all_labels:
	# 	if label_id in label_to_cluster:
	# 		lookup[label_id] = label_to_cluster[label_id]
	# 	else:
	# 		lookup[label_id] = label_id  # Keep original or assign to a default cluster

	# classified_map = lookup[labels]
	return classified_map

def classify_and_score_labels(max_clusters,image,labels):

	inertia = []
	sil_scores = []

	props = regionprops_table(
		labels, 
		intensity_image=image, # standardized_image
		properties=['label', 'mean_intensity']
		)

	labels_features =  np.array([props['mean_intensity-0'], props['mean_intensity-1'], props['mean_intensity-2']]).T
	labels_ids = props['label']

	_,_,C = image.shape
	
	
	for clusters in range(1,max_clusters+1):
		print(f'\nComputing k-means for {clusters}/{max_clusters} clusters')

		initial_centers = np.random.rand(clusters,C)

		kmeans = KMeans(n_clusters=clusters, random_state=42)
		cluster_assignments = kmeans.fit_predict(labels_features)

		label_to_cluster = dict(zip(labels_ids, cluster_assignments))

		unique_labels = np.array(list(label_to_cluster.keys()))

		mapped_clusters = np.array([label_to_cluster[l] for l in unique_labels])

		lookup = np.zeros(labels.max() + 1, dtype=int)
		lookup[unique_labels] = mapped_clusters

		classified_map = 1+lookup[labels]

		centers = kmeans.cluster_centers_

		inertia.append(kmeans.inertia_)

		if clusters > 1:
			sil = silhouette_score(labels_features, cluster_assignments, sample_size=min(1024, len(labels_features)))
			sil_scores.append(sil)

	return inertia, sil_scores

def plot_inertia(inertia,figure_path):
	fig, ax = plt.subplots()
	ax.plot([i for i in range(1,len(inertia)+1)],inertia)

	# Force both axes to show only integer tick labels
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.yaxis.set_major_locator(MaxNLocator(integer=True))
	ax.set_xlabel('Number of clusters (k)', fontsize=12, fontweight='bold')
	ax.set_ylabel('Sum of WCSS', fontsize=12, fontweight='bold')
	ax.set_title('K-means Inertia Plot', fontsize=14, fontweight='bold', pad=15)

	ax.legend(("WCSS", "elbow")) 
	ax.grid(True, which='both', linestyle='--', alpha=0.5)

	plt.savefig(
		figure_path,
		dpi=300,
		transparent=False,
		bbox_inches='tight',
		pad_inches=0.1)
	
def plot_silhouette(sil_scores,figure_path):

	# Plot the results
	plt.figure(figsize=(6,4))
	plt.plot(range(2,2+len(sil_scores)), sil_scores, marker='o')
	plt.xlabel('Number of clusters (k)')
	plt.ylabel('Mean of Silhouette Scores')
	plt.title('K-means Silhouette Score')

	plt.grid(True, ls='--', alpha=0.5)

	plt.savefig(
		figure_path,
		dpi=300,
		transparent=False,
		bbox_inches='tight',
		pad_inches=0.1
		)

def labels_to_geojson(labels, return_dissolved=False, bg_label=0):

	shapes, values = zip(*[[shapely.geometry.shape(s), v] for s, v in features.shapes(labels, labels>bg_label)])

	# Convert to GeoDataFrame. CRS must be None because the coordinates are in pixel space, not a geographic coordinate system.
	# If you don't specify crs=None, it defaults to EPSG:4326, which will lead to errors.
	# You can also add any attributes you want to the GeoDataFrame (it's pandas DataFrame) but it needs geometry.
	
	gdf = gpd.GeoDataFrame({"geometry": shapes, "classification": np.array(values).astype(str), "value": values}, crs=None)

	if return_dissolved:
		gdf_dissolved = gdf.dissolve(by='value', aggfunc='first').reset_index()
		return gdf, gdf_dissolved
	else:
		return gdf