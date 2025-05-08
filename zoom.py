import json
import numpy as np
from PIL import Image
import cv2
import scipy
from scipy import signal
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import scipy.ndimage

'''
	Helper Function - Do Not Modify
	You can use this helper function in generate_warp
'''

def interp2(v, xq, yq):
    dim_input = 1
    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()

    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise('query coordinates Xq Yq should have same shape')

    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor < 0] = 0
    y_floor[y_floor < 0] = 0
    x_ceil[x_ceil < 0] = 0
    y_ceil[y_ceil < 0] = 0

    x_floor[x_floor >= w-1] = w-1
    y_floor[y_floor >= h-1] = h-1
    x_ceil[x_ceil >= w-1] = w-1
    y_ceil[y_ceil >= h-1] = h-1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

    if dim_input == 2:
        return interp_val.reshape(q_h, q_w)
    return interp_val


'''
  Helper function (run me)
  DO NOT MODIFY
'''

def save_gif(morphed_image, fname, fps=6):
    res_list = []
    k = 0
    while k < morphed_image.shape[0]:
        res_list.append(morphed_image[k, :, :, :].astype(np.uint8))
        k += 1

    #   imageio.mimsave(fname, res_list, loop=0)
    imageio.mimsave(fname, res_list, format='GIF', fps=fps, loop=0)


'''
Function - Modify
'''

def overall_bounding_box(H, W, rects, y_padding, x_padding):
    min_y = float("inf")
    min_x = float("inf")
    max_y = float("-inf")
    max_x = float("-inf")


    for rect in rects:
        if rect[0] < min_y:
            min_y = rect[0]
        if rect[1] < min_x:
            min_x = rect[1]
        if rect[2] > max_y:
            max_y = rect[2]
        if rect[3] > max_x:
            max_x = rect[3]

    # TODO: padding should be relative to the scale of the bounding box I think

    min_y -= y_padding
    min_x -= x_padding
    max_y += y_padding
    max_x += x_padding


    min_y = np.clip(min_y, 0, H)
    min_x = np.clip(min_x, 0, W)
    max_y = np.clip(max_y, 0, H)
    max_x = np.clip(max_x, 0, W)

    return min_y, min_x, max_y, max_x

def bounding_zoom(img_H, img_W, H_out, W_out, rects, y_padding, x_padding):

    HW_ratio = H_out / W_out

    min_y, min_x, max_y, max_x = overall_bounding_box(img_H, img_W, rects, y_padding, x_padding)

    y_delta = max_y - min_y
    x_delta = max_x - min_x

    y_mid = (min_y + max_y) * 0.5
    x_mid = (min_x + max_x) * 0.5

    zoom = 1.0

    # If the height of the bounding box is dominant
    if y_delta/x_delta >= HW_ratio:
        # print("height dominant")
        zoom = H_out / y_delta

        # Adjust the width start (min_x)
        new_x_delta = W_out / zoom

        # Width exceeds image
        if new_x_delta >= img_W:
            x_mid = img_W * 0.5
            min_x = x_mid - (new_x_delta / 2)
            # max_x = x_mid + (new_x_delta / 2)

        else:
            min_x = x_mid - (new_x_delta / 2)
            max_x = x_mid + (new_x_delta / 2)

            #Since width does not exceed the image, only one of these can be true
            if min_x < 0:
                min_x = 0

            elif max_x >= img_W:
                max_x = img_W - 1
                min_x = max_x - new_x_delta



    # If the width of the bounding box is dominant
    else:
        # print("width dominant")
        zoom = W_out / x_delta

        # Adjust the height start (min_y)
        new_y_delta = H_out / zoom

        # Height exceeds image
        if new_y_delta >= img_H:
            y_mid = img_H * 0.5
            min_y = y_mid - (new_y_delta / 2)
            # max_x = x_mid + (new_x_delta / 2)

        else:
            min_y = y_mid - (new_y_delta / 2)
            max_y = y_mid + (new_y_delta / 2)

            #Since width does not exceed the image, only one of these can be true
            if min_y < 0:
                min_y = 0

            elif max_y >= img_H:
                max_y = img_H - 1
                min_y = max_y - new_y_delta




    # half_width = ((img_W / zoom) / 2.0)
    # half_height = ((img_H / zoom) / 2.0)

    # if x + half_width > img_W:


    return min_y, min_x, zoom

# Seems to me like higher velocity should mean more interpolation towards the new value
# def interpolate_zoom(y1, x1, zoom1, y2, x2, zoom2, factor = 0.5):
#     inv_factor = 1.0 - factor
#     return y1 * inv_factor + y2 * factor, x1 * inv_factor + x2 * factor, zoom1 * inv_factor + zoom2 * factor

def interpolate_zoom(y1, x1, zoom1, y2, x2, zoom2, y_interp, x_interp, zoom_interp):
    # inv_factor = 1.0 - factor
    return y1 * (1.0-y_interp) + y2 * y_interp, x1 * (1-x_interp) + x2 * x_interp, zoom1 * (1-zoom_interp) + zoom2 * zoom_interp


def get_interp_vals(y1, x1, zoom1, y2, x2, zoom2, y_interp_in, x_interp_in, zoom_interp_in, ref_vecs_2d, vec_3d_changes):

    y_interp, x_interp, zoom_interp = y_interp_in, x_interp_in, zoom_interp_in

    # av_ref_2d = np.array([0.0,0.0])
    # for ref_vec in ref_vecs_2d:
    #     np_vec = np.array(ref_vec)
    #     if (np.linalg.norm(np_vec) > 0):
    #         av_ref_2d += np_vec / np.linalg.norm(np_vec)
    # av_ref_2d /= len(ref_vecs_2d)

    av_ref_2d = np.array([0.0,0.0])
    for ref_vec in ref_vecs_2d:
        np_vec = np.array(ref_vec)
        av_ref_2d += np_vec / np.linalg.norm(np_vec)
    # print(av_ref_2d)
    if (np.linalg.norm(av_ref_2d) > 0):
        av_ref_2d /= np.linalg.norm(av_ref_2d)
    # print(av_ref_2d)

    av_vec_3d_change = 0
    for vec_3d_change in vec_3d_changes:
        av_vec_3d_change += vec_3d_change
    av_vec_3d_change /= len(vec_3d_changes)

    # print(av_vec_3d_change)

    # av_vec_3d_change += 1.0
    # av_vec_3d_change /= 2.0

    # print(av_vec_3d_change)


    box_2d_vel = np.array([x1 - x2, y1 - y2])
    if (np.linalg.norm(box_2d_vel) > 0):
        box_2d_vel = box_2d_vel / np.linalg.norm(box_2d_vel)
    else:
        box_2d_vel = av_ref_2d


    ref_comp_vel = av_ref_2d @ box_2d_vel
    # print(ref_comp_vel)

    ref_comp_vel += 1.0
    ref_comp_vel /= 2.0

    # print(ref_comp_vel)

    ref_comp_vel -= 1.0
    ref_comp_vel = abs(ref_comp_vel)
    # print(ref_comp_vel)


    # y_interp, x_interp, zoom_interp = max(0.1, y_interp * ref_comp_vel[1]), max(0.1, x_interp * ref_comp_vel[0]), max(0.1, zoom_interp * av_vec_3d_change)

    # y_interp, x_interp, zoom_interp = max(y_interp, ref_comp_vel[1]), max(x_interp, ref_comp_vel[0]), max(zoom_interp, av_vec_3d_change)

    # y_interp, x_interp, zoom_interp = max(0.2, ref_comp_vel * y_interp), max(0.2, ref_comp_vel * x_interp), max(0.2, min(zoom_interp, av_vec_3d_change))

    # y_interp, x_interp, zoom_interp = max(y_interp, ref_comp_vel * 0.95), max(x_interp, ref_comp_vel * 0.95), max(0.2, min(zoom_interp, av_vec_3d_change))

    change = ((ref_comp_vel + av_vec_3d_change) / 2.0) * 0.95
    # print(change)

    y_interp, x_interp, zoom_interp = max(y_interp, change), max(x_interp, change), max(zoom_interp, change)

    # y_interp, x_interp, zoom_interp = change, change, change


    # print(y_interp, x_interp, zoom_interp)

    # y_interp, x_interp, zoom_interp = y_interp_in, x_interp_in, zoom_interp_in


    return y_interp, x_interp, zoom_interp


def get_yx_min(H_img, W_img, y, x, zoom):
    return (y - (H_img / zoom) / 2.0), (x - (W_img / zoom) / 2.0)

def next_frame_zoom(H, W, y_min, x_min, zoom, image):
    x_mesh, y_mesh = np.meshgrid(np.arange(0, W, 1), np.arange(0, H, 1))

    xy_mesh = np.stack((x_mesh, y_mesh), axis=2)



    # addition = np.tile(np.array([x_min, y_min]), (3, 1))

    xy_on_image = xy_mesh / zoom + np.array([x_min, y_min])

    scale_mat = np.identity(2) * zoom

    # result_image = image

    # Generate Warped Images (Use function interp2) for each of 3 layers
    result_image = np.zeros((H, W, 3), dtype=np.uint8)

    for i in range(3):
        result_image[:,:,i] = interp2(image[:,:,i], xy_on_image[:,:,0], xy_on_image[:,:,1]).reshape(H, W)


    return result_image


# def comp_pos(annotation1, annotation2):
#     return (annotation1["x"] - annotation2["x"]) ** 2 + (annotation1["y"] - annotation2["y"]) ** 2 






def zoom_all(parsed_data, use_pose, selected_ids, selected_labels, nearest_to_ball = 0, x_padding = 10, y_padding = 10, x_interp = 0.5, y_interp = 0.5, zoom_interp = 0.5, H_out = 0, W_out = 0):

    results = None

    i = 0

    prev_y_min, prev_x_min, prev_zoom = -1, -1, -1

    for i in range(len(parsed_data)):
        image_name, annotations = parsed_data[i]

        rects = []

        if (nearest_to_ball > 0):

            lower_annotations = annotations.copy()

            for annotation in annotations:
                label = annotation["label"]
                if (label == "basketball"):
                    basketball_annotation = annotation

                    lower_annotations = sorted(lower_annotations, key = lambda x :
                        ((x["x"] - basketball_annotation["x"]) ** 2 + (x["y"] - basketball_annotation["y"]) ** 2)
                    )

                    break

            j = 0
            for annotation in lower_annotations:
                id = annotation["id"]
                label = annotation["label"]

                x = annotation["x"]
                y = annotation["y"]
                width = annotation["width"]
                height = annotation["height"]

                if (label != "basketball") and (label == "player") and (j < nearest_to_ball):
                    rects.append([y - height/2, x - width/2, y + height/2, x + width/2])
                    j += 1
                if j >= 3:
                    break

        ref_vecs_2d = []
        vec_3d_changes = []

        for annotation in annotations:
        # for j in range(len(annotations)):
            # annotation = annotations[j]

            id = annotation["id"]
            label = annotation["label"]
            # coordinates = annotation["coordinates"]

            x = annotation["x"]
            y = annotation["y"]
            width = annotation["width"]
            height = annotation["height"]

            if "facing_direction_2d" in annotation:

                facing_direction_2d = annotation["facing_direction_2d"]
                facing_direction_3d = annotation["facing_direction_3d"]
                # position_2d = annotation["position_2d"]


                ref_vecs_2d.append(facing_direction_2d)

                # facing_direction_3d = np.array(annotation["facing_direction_3d"])
                # facing_direction_3d = facing_direction_3d / np.linalg.norm(facing_direction_3d)

                # facing_direction_3d_prev = []
                # if i > 0:
                #     facing_direction_3d_prev = np.array(parsed_data[i-1][1][j]["facing_direction_3d"])
                # else:
                #     facing_direction_3d_prev = facing_direction_3d
                
                # facing_direction_3d_prev = facing_direction_3d_prev / np.linalg.norm(facing_direction_3d_prev)
                # vec_3d_changes.append(facing_direction_3d @ facing_direction_3d_prev)

                vec_3d_changes.append(annotation["3d_change"])

            # if (label == "basketball" or label == "referee"):
            #     rects.append([y - height/2, x - width/2, y + height/2, x + width/2])

            # if (label == "basketball"):
            #     rects.append([y - height/2, x - width/2, y + height/2, x + width/2])

            if (label in selected_labels or id in selected_ids):
                rects.append([y - height/2, x - width/2, y + height/2, x + width/2])

            # rects.append([y - height/2, x - width/2, y + height/2, x + width/2])



        image = Image.open(f'output_video_frames/{image_name}')
        image_ar = np.array(image)

        img_H, img_W, _ = image_ar.shape

        # TODO: option here
        if H_out == 0:
            # H_out = int(img_H / 2)
            H_out = img_H
        if W_out == 0:
            # W_out = int(img_W / 2)
            W_out = img_W

        if results is None:
            results = np.zeros((len(parsed_data), H_out, W_out, 3), dtype=np.uint8)

        # y_padding, x_padding = 10,10
        # y_padding, x_padding = 70,70

        y_min, x_min, zoom = bounding_zoom(img_H, img_W, H_out, W_out, rects, y_padding, x_padding)

        # If on the first loop
        if prev_zoom == -1:
            prev_y_min, prev_x_min, prev_zoom = y_min, x_min, zoom

        if use_pose:
            new_y_interp, new_x_interp, new_zoom_interp = get_interp_vals(y_min, x_min, zoom, prev_y_min, prev_x_min, prev_zoom, y_interp, x_interp, zoom_interp, ref_vecs_2d, vec_3d_changes)

            y_min, x_min, zoom = interpolate_zoom(y_min, x_min, zoom, prev_y_min, prev_x_min, prev_zoom, new_y_interp, new_x_interp, new_zoom_interp)

        else:
            y_min, x_min, zoom = interpolate_zoom(y_min, x_min, zoom, prev_y_min, prev_x_min, prev_zoom, y_interp, x_interp, zoom_interp)


        res = next_frame_zoom(H_out, W_out, y_min, x_min, zoom, image_ar)
        # plt.imshow(res)
        # plt.show()

        results[i] = res

        i += 1
        prev_y_min, prev_x_min, prev_zoom = y_min, x_min, zoom

    return results


def error_correction_json(parsed_data):

    out_data = []

    last_id_pos = {}
    last_id_vels = {}
    last_id_3d = {}
    last_id_annotations = {}

    # For every image
    for i in range(len(parsed_data)):
        image_name, annotations = parsed_data[i]

        out_data.append((image_name, []))

        cur_ids = set()
        # id_to_label = {}

        # For every annotation in the images' annotations
        for annotation in annotations:
            out_annotation = annotation

            label = annotation["label"]
            id = annotation["id"]
            x = annotation["x"]
            y = annotation["y"]
            width = annotation["width"]
            height = annotation["height"]

            

            annotation_data = {}

            facing_direction_2d = []
            facing_direction_3d = []
            # position_2d = []

            if "facing_direction_2d" in annotation:
                facing_direction_2d = annotation["facing_direction_2d"]
                facing_direction_3d = annotation["facing_direction_3d"]
                # position_2d = annotation["position_2d"]

                annotation_data = {
                    "label": label,
                    "id": id,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "facing_direction_2d": facing_direction_2d,
                    "facing_direction_3d": facing_direction_3d,
                    "3d_change": 0.0
                    # "position_2d": position_2d
                }
            
            else:
                annotation_data = {
                    "label": label,
                    "id": id,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }


            

            out_data[i][1].append(annotation_data)


            cur_ids.add(id)

            # If the id wasn't found yet
            if id not in last_id_pos:
                last_id_vels[id] = (0, 0)
                last_id_pos[id] = (x, y)
                last_id_annotations[id] = annotation_data

                if "facing_direction_2d" in annotation:
                    last_id_3d[id] = facing_direction_3d


                #TODO: add annotation for previous frames if there was a prev frame
                for j in range(i):
                    image_name_prev, annotations_prev = parsed_data[j]

                    prev_annotation_data = annotation_data.copy()

                    out_data[j][1].append(prev_annotation_data)


            # If the id was already found
            else:
                x_last, y_last = last_id_pos[id]
                x_vel, y_vel = x - x_last, y - y_last
                
                if "facing_direction_2d" in annotation:
                    facing_direction_3d = np.array(facing_direction_3d)
                    facing_direction_3d = facing_direction_3d / np.linalg.norm(facing_direction_3d)

                    facing_direction_3d_prev = last_id_3d[id]
                    facing_direction_3d_prev = facing_direction_3d_prev / np.linalg.norm(facing_direction_3d_prev)

                    annotation_data["3d_change"] = (facing_direction_3d @ facing_direction_3d_prev)

                    last_id_3d[id] = facing_direction_3d

                last_id_pos[id] = (x, y)
                last_id_vels[id] = (x_vel, y_vel)
                last_id_annotations[id] = annotation_data

        # For every id detected so far
        for id in last_id_pos:

            # If the id wasn't detected for this image
            if id not in cur_ids:

                last_annotation = last_id_annotations[id]

                x, y = last_id_pos[id]
                x_vel, y_vel = last_id_vels[id]
                new_x, new_y = x + x_vel, y + y_vel

                new_annotation = last_annotation.copy()
                new_annotation["x"] = new_x
                new_annotation["y"] = new_y

                out_data[i][1].append(new_annotation)

    return out_data



def parse_json(file_path):

    # file_path = base_path + file_path

    with open(file_path, 'r') as f:
        data = json.load(f)

    parsed_data = []
    # i = 0
    for entry in data:
        image_name = entry["image"]
        # image_name = base_path + entry["image"]
        annotations = []

        for annotation in entry["annotations"]:
            label = annotation["label"]
            id = annotation["id"]
            coordinates = annotation["coordinates"]

            annotation_data = {}

            if "facing_direction_2d" in annotation:
                annotation_data = {
                    "label": label,
                    "id": id,
                    "x": coordinates["x"],
                    "y": coordinates["y"],
                    "width": coordinates["width"],
                    "height": coordinates["height"],
                    "facing_direction_2d": annotation["facing_direction_2d"],
                    "facing_direction_3d": annotation["facing_direction_3d"],
                    "position_2d": annotation["position_2d"]
                }
            
            else:
                annotation_data = {
                    "label": label,
                    "id": id,
                    "x": coordinates["x"],
                    "y": coordinates["y"],
                    "width": coordinates["width"],
                    "height": coordinates["height"]
                }

            # annotation_data = {
            #     "label": label,
            #     "id": id,
            #     "x": annotation["x"],
            #     "y": annotation["y"],
            #     "width": annotation["width"],
            #     "height": annotation["height"],
            #     "facing_direction_2d": annotation["facing_direction_2d"],
            #     "facing_direction_3d": annotation["facing_direction_3d"],
            #     "position_2d": annotation["position_2d"]
            # }

            annotations.append(annotation_data)

        parsed_data.append((image_name, annotations))
        # i += 1

    return parsed_data



def zoom_save_gif(out_file, use_pose, fps, selected_ids, selected_labels, nearest_to_ball, x_padding, y_padding, x_interp, y_interp, zoom_interp):
    # base_path = 'C:/Users/justi/Documents/GitHub/CIS581-Final-Project/'

    
    file_path = 'output/annot.json'  # Path to your JSON file
    # parsed_data = parse_json(base_path, file_path)
    parsed_data = parse_json(file_path)
    corrected_data = error_correction_json(parsed_data)
    # results = zoom_all(corrected_data, base_path, use_pose, selected_ids, selected_labels, nearest_to_ball, x_padding, y_padding, x_interp, y_interp, zoom_interp)
    results = zoom_all(corrected_data, use_pose, selected_ids, selected_labels, nearest_to_ball, x_padding, y_padding, x_interp, y_interp, zoom_interp)
    save_gif(results, out_file, fps)
    return out_file


# zoom_save_gif("BasketZoomHalfRes3.gif", 12, set(), {"basketball"}, 3, x_padding=20, y_padding=20, x_interp=0.5, y_interp=0.5, zoom_interp=0.5)
