import cv2
import pdb
import numpy as np
import click
import shutil
import json
import time
from tqdm import tqdm
from pathlib import Path
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon


SCRIPT_DIR = str(Path(__file__).parent)
POLYGON_TEMPLATE_JSON = "./scripts/polygon_template.json"


def generate_pseudo_meta_data():
    timestamp = int(time.time_ns() / 1000000)
    return {"type": "meta", "name": "lastAction", "timestamp": timestamp}


def polygon_mask(poly, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    int_coords = lambda x: np.array(x).round().astype(np.int32)[:, [1, 0]]
    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    img_mask = cv2.fillPoly(img_mask, exteriors, 255)
    img_mask = cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def add_cumulative_mask(mask, polygon):
    mask_tmp = np.zeros(mask.shape, dtype=np.uint8)
    poly_as_contour = np.asarray(polygon.exterior.coords)
    mask_tmp = polygon_mask(polygon, mask.shape)
    mask = cv2.bitwise_or(mask_tmp, mask)
    return mask


def create_sub_mask_annotation(sub_mask, width, height, kernel_size=10, draw_cumulative_polygon_mask=False):
    # Remove noise mask element
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, kernel)

    # Get Whole mask
    mask_tmp = np.zeros([height, width], dtype=np.uint8)
    contours, _ = cv2.findContours(sub_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cumulative_mask = np.zeros((height, width), dtype=np.uint8)

    polygons = []
    segmentations = []
    j = 0
    for contour in contours:
        contour = contour[:, 0, :]

        # Remove small mask region
        if cv2.contourArea(contour[:, np.newaxis, :]) < 10000:
            continue

        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        n_vertice = np.array(poly.exterior.coords).shape[0]
        poly = poly.simplify(1.0, preserve_topology=False)
        if poly.is_empty:
            # Go to next iteration, dont save empty values in list
            continue
        if type(poly) == MultiPolygon:
            continue
        else:
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            polygons.append(poly)
            cumulative_mask = add_cumulative_mask(cumulative_mask, poly)
            segmentations.append(segmentation)

    if draw_cumulative_polygon_mask:
        return polygons, segmentations, cumulative_mask
    else:
        return polygons, segmentations, None


def segmentation_to_polygon_obj(segmentation, class_id=1):
    polygon_template_json_path = str(Path(POLYGON_TEMPLATE_JSON))
    with open(polygon_template_json_path) as f:
        polygon_template_json = json.load(f)
    polygon_template_json["classId"] = class_id
    polygon_template_json["points"] = segmentation[0]
    return polygon_template_json


def write_cumulative_mask(cum_mask, cum_mask_dir, base_name):
    cum_mask_save_path = str(Path(cum_mask_dir, base_name))
    cv2.imwrite(cum_mask_save_path, cum_mask)
    cv2.waitKey(10)


def get_polygon_object_list(polygons, class_id):
    polygon_object_list = []
    for polygon in polygons:
        if type(polygon) == MultiPolygon:
            print("MultiPolygon Detected")
            pass
            '''
            n_polygon = len(polygon.geoms)
            for i in range(n_polygon):
                polygon_sub = polygon.geoms[i]
                segmentation = [np.array(polygon_sub.exterior.coords).ravel().tolist()]
                polygon_obj = segmentation_to_polygon_obj(segmentation, class_id=class_id)
                polygon_object_list.append(polygon_obj)
            '''
        else:
            segmentation = [np.array(polygon.exterior.coords).ravel().tolist()]
            polygon_obj = segmentation_to_polygon_obj(segmentation, class_id=class_id)
            polygon_object_list.append(polygon_obj)
    return polygon_object_list


@click.command()
@click.option("--mask-dir", "-c", default=f"{SCRIPT_DIR}/mask")
@click.option("--cum-mask-dir", "-c", default=f"{SCRIPT_DIR}/mask_cum")
@click.option("--width", "-w", default=1920)
@click.option("--height", "-h", default=1080)
@click.option("--save-cumulative-mask", is_flag=True)
def main(mask_dir, cum_mask_dir, width, height, save_cumulative_mask):
    mask_pathes = Path(mask_dir).glob("*.png")
    mask_path_list = [str(mask_path) for mask_path in mask_pathes]
    mask_path_list = np.sort(mask_path_list)

    default_class_id = 1
    if not Path(cum_mask_dir).exists():
        Path(cum_mask_dir).mkdir()

    output_json_obj = {}
    for mask_path in tqdm(mask_path_list):
        base_name = Path(mask_path).name
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
        mask = cv2.resize(mask, (width, height), cv2.INTER_NEAREST)

        # Convert mask image to json dictionary
        polygons, segmentations, cum_mask = create_sub_mask_annotation(mask, width, height, draw_cumulative_polygon_mask=save_cumulative_mask)
        if save_cumulative_mask:
            write_cumulative_mask(cum_mask, cum_mask_dir, base_name)

        polygon_obj_list = get_polygon_object_list(polygons, default_class_id)

        meta_obj = generate_pseudo_meta_data()
        polygon_obj_list.append(meta_obj)
        output_json_obj[f"{base_name}"] = polygon_obj_list

    with open("./annotation.json", "w") as f:
        json.dump(output_json_obj, f)


if __name__ == "__main__":
    main()
