import os
import json
import shutil
from tqdm import tqdm
import yaml
import cv2


# Convert between coco json and yolo format

def coco2json(model_cfg):
    data_root = f"./datasets/{model_cfg['task']}/{model_cfg['dataset_name']}/"
    for root, folders, files in os.walk(data_root):
        if "coco" in root:
            if "train.json" in files:
                nc_list = [] # Collect list of classes
                yolo_dir = root.replace("coco", "yolo")
                os.makedirs(os.path.join(yolo_dir, "train", "images"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dir, "train", "labels"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dir, "valid", "images"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dir, "valid", "labels"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dir, "test", "images"), exist_ok=True)
                os.makedirs(os.path.join(yolo_dir, "test", "labels"), exist_ok=True)

                ##### Go through train.json, val.json, test.json
                for dataset_json in ["train", "val", "test"]:
                    dataset_json_fpath = os.path.join(root, dataset_json)
                    with open(dataset_json_fpath + ".json", "r") as f:
                        dataset_json_dict = json.load(f)
                        id_sorted_ims = sorted(dataset_json_dict["images"], key=lambda im: im["id"])
                    if dataset_json == "val":
                        dataset_json = "valid"
                    ##### Go though the "images" and "annotations" keys
                    with tqdm(iterable=id_sorted_ims) as tq:
                        for im in id_sorted_ims:
                            coco_image_fpath = os.path.join(root, "images", im["file_name"])
                            yolo_image_fpath = os.path.join(yolo_dir, dataset_json, "images", im["file_name"])
                            coco_label_fpath = os.path.join(root, "annotations", im["file_name"].replace("jpg", "json"))
                            yolo_label_fpath = os.path.join(yolo_dir, dataset_json, "labels", im["file_name"].replace("jpg", "txt"))

                            ##### Copy image to corresponding images folder
                            try:
                                shutil.copy(coco_image_fpath, yolo_image_fpath)
                            except Exception as e:
                                print(e)
                                continue

                            ##### Create txt file of same name in a labels folder
                            with open(coco_label_fpath, "r") as coco_f:
                                coco_label_dict = json.load(coco_f)
                            
                            img_wh = [coco_label_dict["imageWidth"], coco_label_dict["imageHeight"]] # Use this to normalize labels
                            yolo_label_list = []
                            for shape in coco_label_dict["shapes"]: # for each instance
                                if shape["label"] not in nc_list:
                                    nc_list.append(shape["label"]) # add the class to our list if we haven't seen it before
                                label_ind = nc_list.index(shape["label"])
                                points = [str(label_ind)]
                                for pt in shape["points"]:
                                    points.extend([str(pt[0] / img_wh[0]), str(pt[1] / img_wh[1])])
                                yolo_label_list.append(" ".join(points))
                            with open(yolo_label_fpath, "w") as yolo_f:
                                yolo_f.write("\n".join(yolo_label_list))
                            tq.update()
                dataset_yaml = {
                    "nc": len(nc_list),
                    "names": nc_list,
                    "path": yolo_dir.replace("./datasets/", "./"),
                    "train": "train/images",
                    "val": "valid/images",
                    "test": "test/images"
                }
                with open(os.path.join(yolo_dir, "data.yaml"), "w") as yolo_f:
                    yaml.dump(dataset_yaml, yolo_f)
    # Create data.yaml file
    # We've now ensured we have a data.yaml file, let's ensure that the image size is accounted for
    

if __name__ == "__main__":
    from utils import initialize_wandb
    model_cfg, wandb_cfg = initialize_wandb("./oak_segmentation/model_cfg.json")
    coco2json(wandb_cfg)