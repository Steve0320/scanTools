# Script for automating a basic photogrammetry pipeline
# Expects to be given a path to a folder containing images to use

import time
import glob
import os, os.path
import exiftool
from tqdm import tqdm
from pathlib import Path
from backgroundremover import utilities
from backgroundremover.bg import remove, get_model, detect, naive_cutout, DEVICE
import Metashape
from PIL import Image
import io
import numpy as np


# Helper - run background remover
def remove_background(in_dir, out_dir):

	print(DEVICE)

	model = get_model("u2net")

	for name in tqdm(input_files):

		input_path = os.path.join(in_dir, name)
		output_path = os.path.join(out_dir, os.path.splitext(name)[0] + '.png')

		with open(input_path, 'rb') as f:
			image_data = f.read()

		## BEGIN COPY
		img = Image.open(io.BytesIO(image_data)).convert("RGB")
		mask = detect.predict(model, np.array(img)).convert("L")

		cutout = naive_cutout(img, mask)

		bio = io.BytesIO()
		cutout.save(bio, "PNG")

		output = bio.getbuffer()

		## END COPY

		with open(output_path, 'wb') as output_file:
			output_file.write(output)


def absolute_file_paths(directory):
	for dirpath, _, filenames in os.walk(directory):
		for f in filenames:
			yield os.path.abspath(os.path.join(dirpath, f))


def process_metashape(photo_dir, proj_dir):

	doc = Metashape.Document()
	doc.save(os.path.join(proj_dir, "project.psx"))
	chunk = doc.addChunk()

	print(list(absolute_file_paths(photo_dir)))
	chunk.addPhotos(list(absolute_file_paths(photo_dir)))
	doc.save()

	chunk.matchPhotos(
		keypoint_limit=40000,
		tiepoint_limit=4000,
		generic_preselection=True,
		reference_preselection=True
	)
	doc.save()

	chunk.alignCameras()
	chunk.buildDepthMaps(downscale=4, filter_mode=Metashape.AggressiveFiltering)

	chunk.buildModel(
		source_data=Metashape.DepthMapsData,
		surface_type=Metashape.Arbitrary,
		interpolation=Metashape.EnabledInterpolation
	)

	chunk.buildUV(mapping_mode=Metashape.GenericMapping)
	chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=8192)
	doc.save()


def copy_exif(input_list, src_dir, dest_dir):
	with exiftool.ExifTool() as et:
		for name in tqdm(input_list):
			ip = os.path.join(src_dir, name)
			op = os.path.join(dest_dir, os.path.splitext(name)[0] + '.png')
			et.execute("-overwrite_original", "-TagsFromFile", ip, op)


if __name__ == '__main__':

	input_dir = "/home/steve/PycharmProjects/scanTools/GoblinRaw"
	output_dir = "/home/steve/PycharmProjects/scanTools/Goblin1"

	masked_dir_name = "removed_bg"
	masked_dir = os.path.join(output_dir, masked_dir_name)

	valid_input_formats = [".jpg", ".png"]

	# Ensure output directory exists
	Path(output_dir).mkdir(exist_ok=True)
	Path(masked_dir).mkdir(exist_ok=True)

	# Find all input files
	input_files = []
	for f in os.listdir(input_dir):
		ext = os.path.splitext(f)[1]
		if ext.lower() not in valid_input_formats:
			continue
		input_files.append(f)

	# TODO: Parallelize
	# Remove image backgrounds
	print(f"Removing backgrounds for {len(input_files)} files...")
	remove_background(input_dir, masked_dir)

	# Copy camera EXIF data to trimmed images
	print("Copying EXIF data...")
	copy_exif(input_files, input_dir, output_dir)

	# Process in metashape
	print("Processing in Metashape")
	process_metashape(masked_dir, output_dir)