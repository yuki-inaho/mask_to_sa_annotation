import cv2
import numpy as np
import click
from pathlib import Path
from skimage import measure
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon
import pdb
import shutil
import json
import time


SCRIPT_DIR = str(Path(__file__).parent)
POLYGON_TEMPLATE_JSON = "./scripts/polygon_template.json"


def generate_pseudo_meta_data():
    timestamp = int(time.time_ns() / 1000000)
    return {"type":"meta","name":"lastAction","timestamp":timestamp}


def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation="low")

    polygons = []
    segmentations = []
    j = 0
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        n_vertice = np.array(poly.exterior.coords).shape[0]
        poly = poly.simplify(1.0, preserve_topology=False)
        #simplification_rate = np.max([1.0, n_vertice/100.0])
        #poly = poly.simplify(simplification_rate, preserve_topology=True)
        poly = poly.simplify(0.1, preserve_topology=True)

        if poly.is_empty:
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)

        if type(poly) == MultiPolygon:
            n_poly = len(poly.geoms)
            for i in range(n_poly):
                poly_sub = poly.geoms[i]
                segmentation = np.array(poly_sub.exterior.coords).ravel().tolist()
                segmentations.append(segmentation)
        else:
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)

    return polygons, segmentations


def segmentation_to_polygon_obj(segmentation, class_id=1):
    polygon_template_json_path = str(Path(POLYGON_TEMPLATE_JSON))
    with open(polygon_template_json_path) as f:
        polygon_template_json = json.load(f)
    polygon_template_json['classId'] = class_id
    polygon_template_json['points'] = segmentation[0]

    return polygon_template_json


@click.command()
@click.option("--mask-dir", "-c", default=f"{SCRIPT_DIR}/mask")
@click.option("--width", "-w", default=1920)
@click.option("--height", "-h", default=1080)
def main(mask_dir, width, height):
    mask_pathes = Path(mask_dir).glob("*.png")
    mask_path_list = [str(mask_path) for mask_path in mask_pathes]
    mask_path_list = np.sort(mask_path_list)
    default_class_id = 1

    output_json_obj = {}
    for mask_path in tqdm(mask_path_list):
        base_name = Path(mask_path).name
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
        mask = cv2.resize(mask, (width, height), cv2.INTER_NEAREST)
        polygons, segmentations = create_sub_mask_annotation(mask)

        polygon_obj_list = []
        for k, polygon in enumerate(polygons):
            if type(polygon) == MultiPolygon:
                n_polygon = len(polygon.geoms)
                for i in range(n_polygon):
                    polygon_sub = polygon.geoms[i]
                    segmentation = [np.array(polygon_sub.exterior.coords).ravel().tolist()]
                    polygon_obj = segmentation_to_polygon_obj(segmentation, class_id=default_class_id)
                    polygon_obj_list.append(polygon_obj)
            else:
                segmentation = [np.array(polygon.exterior.coords).ravel().tolist()]
                polygon_obj = segmentation_to_polygon_obj(segmentation, class_id=default_class_id)
                polygon_obj_list.append(polygon_obj)

        meta_obj = generate_pseudo_meta_data()
        polygon_obj_list.append(meta_obj)
        output_json_obj[f"{base_name}"] = polygon_obj_list

    with open('./annotation.json', 'w') as f:
        json.dump(output_json_obj, f)

if __name__ == "__main__":
    main()

