from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
import pdb
import json
import pdb

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def human_pose_estimation(img_folder, out_folder):
    import time
    start = time.time()

    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2
    cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Get all demo images that end with .jpg or .png
    # img_paths = sorted([img for end in args.file_type for img in Path(args.img_folder).glob(end)])
    file_type = ['*.jpg', '*.png']
    img_paths = sorted([img for end in file_type for img in Path(img_folder).glob(end)],key=lambda x: int(''.join(filter(str.isdigit, x.stem))))  # Extract numeric part of filename

    # save json for facing orientations
    facing_direction_whole_dataset = {}

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        det_instances = det_out['instances']
        
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5) 
        boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            t1 = time.time()
            with torch.no_grad():
                out = model(batch)
            t2 = time.time()
            print(t2-t1)
            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
        

            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()
     
                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)


        misc_args = dict(
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        facing_direction_2d, facing_direction_3d, position_2d = renderer.direction_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args)
        

        det_jpg_name = str(img_path).split("/")[-1]
        subdict = {}
        subdict["facing_direction_2d"] = facing_direction_2d
        subdict["facing_direction_3d"] = facing_direction_3d
        subdict["position_2d"] = position_2d
        facing_direction_whole_dataset[det_jpg_name] = subdict

        end = time.time()
        print(end - start)


    # # Save to JSON
    facing_direction_whole_dataset.keys()
    with open(os.path.join(out_folder,"facing_directions.json"), "w") as json_file:
        json.dump(facing_direction_whole_dataset, json_file, indent=4)


def integration(annotations, facing_direction_whole_dataset):
    # Rearrange data
    rearranged_data = {
        entry["image"]: [
            {
                "id": annotation["id"],
                "label": annotation["label"],
                "coordinates": [
                    annotation["coordinates"]["x"],
                    annotation["coordinates"]["y"],
                    annotation["coordinates"]["width"],
                    annotation["coordinates"]["height"],
                ],
            }
            for annotation in entry["annotations"]
        ]
        for entry in annotations
    }



    # Update rearranged_data with matching facing direction and position data
    for key in rearranged_data.keys():
        if key in facing_direction_whole_dataset:
            # Iterate over entries in rearranged_data[key]
            for item in rearranged_data[key]:
                # Get the bounding box coordinates
                coordinates = item["coordinates"]

                # Extract the center point of the bounding box (x_center, y_center)
                x_center = coordinates[0]
                y_center = coordinates[1]

                # Match with the position in the second dictionary
                for i, position in enumerate(facing_direction_whole_dataset[key]["position_2d"]):
                    pos_x, pos_y = position  # Extract position coordinates
                    
                    # Check if the positions are approximately equal
                    if abs(x_center - pos_x) < 10 and abs(y_center - pos_y) < 10:
                        # Add facing direction and position data to the item
                        item["facing_direction_2d"] = facing_direction_whole_dataset[key]["facing_direction_2d"][i]
                        item["facing_direction_3d"] = facing_direction_whole_dataset[key]["facing_direction_3d"][i]
                        item["position_2d"] = position
                        break  # Stop searching after finding a match


    # Create a new dictionary to store matched entries
    matched_data = {}

    # Iterate over keys in rearranged_data
    for key in rearranged_data.keys():
        if key in facing_direction_whole_dataset:
            matched_data[key] = []  # Initialize an empty list for this key

            # Iterate over entries in rearranged_data[key]
            for item in rearranged_data[key]:
                # Get the bounding box coordinates
                coordinates = item["coordinates"]
                
                if item['label'] == 'basketball':
                    matched_item = {
                            "id": item["id"],
                            "label": item["label"],
                            "coordinates" : {
                                "x": coordinates[0],
                                "y": coordinates[1],
                                "width": coordinates[2],
                                "height": coordinates[3],
                            },
                        }
                    matched_data[key].append(matched_item)
                    continue

                # Extract the center point of the bounding box (x_center, y_center)
                x_center = coordinates[0]
                y_center = coordinates[1]

                # Match with the position in the second dictionary
                for i, position in enumerate(facing_direction_whole_dataset[key]["position_2d"]):
                    pos_x, pos_y = position  # Extract position coordinates

                    # Check if the positions are approximately equal
                    if abs(x_center - pos_x) < 10 and abs(y_center - pos_y) < 10:
                        # Create a new matched item
                        matched_item = {
                            "id": item["id"],
                            "label": item["label"],
                            "coordinates" : {
                                "x": coordinates[0],
                                "y": coordinates[1],
                                "width": coordinates[2],
                                "height": coordinates[3],
                            },
                            "facing_direction_2d": facing_direction_whole_dataset[key]["facing_direction_2d"][i],
                            "facing_direction_3d": facing_direction_whole_dataset[key]["facing_direction_3d"][i],
                            "position_2d": position,
                        }
                        matched_data[key].append(matched_item)  # Add to the new dictionary
                        break  # Stop searching after finding a match



    integration_list = []                    
    for key, item in matched_data.items():
        dictFor1img = {
            "image": key,
        }
        dictFor1img["annotations"] = []
        for subdict in item:
            dictFor1img["annotations"].append(subdict)
        integration_list.append(dictFor1img)
    
    return integration_list

def rearrange_data(output_folder):

    annot_path = os.path.join(output_folder,"annot.json")
    with open(annot_path, 'r') as file:
        annotations = json.load(file)
    facing_path = os.path.join(output_folder,"facing_directions.json")
    with open(facing_path, 'r') as file:
        facing_direction_whole_dataset = json.load(file)
    
    annots_and_poses = integration(annotations, facing_direction_whole_dataset)
    annots_and_poses_path = os.path.join(output_folder, "annots_and_poses.json")
    with open(annots_and_poses_path, 'r') as file:
        json.dump(annots_and_poses, file, indent=4)