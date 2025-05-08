import os
import glob
import io
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import json
import cv2
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
from sam2.build_sam import build_sam2_video_predictor
from collections import Counter
import inference
from dotenv import load_dotenv

# Utils

def video_to_frames(video_path, output_dir='output_video_frames'):
    """
    Extracts frames from a video file and saves them as image files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Deletes all current files in output directory
    for file_path in glob.glob(f"{output_dir}/*"):
        os.remove(file_path)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    frame_count = 0
    success, frame = video_capture.read()

    # Loop through all frames
    while success:
        # Save each frame as an image file
        frame_filename = os.path.join(output_dir, f'{frame_count:05d}.jpg')
        cv2.imwrite(frame_filename, frame)

        frame_count += 1
        success, frame = video_capture.read()

    video_capture.release()
    print(f"Extracted {frame_count} frames to '{output_dir}'.")

def select_device():
    """
    Selects available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    
    return device

def extract_integer(filename):
    """
    Returns the frame number of input filename as an integer.
    """
    return int(filename.split(".")[0])

def preprocess_caption(labels):
    """
    Appends a period to the end of each label.
    """
    return [label if label.endswith(".") else label + "." for label in labels]

def combine_results(results):
    """
    Combines scores, labels, and boxes from DINO detections of separate labels.
    """
    combined_scores = []
    combined_labels = []
    combined_boxes = []
    
    for detection in results:
        combined_scores.extend(detection['scores'].tolist())
        combined_labels.extend(detection['labels'])
        combined_boxes.extend(detection['boxes'].tolist())
    
    return combined_scores, combined_labels, combined_boxes

def load_frames(frame_path):
    """
    Returns video frames given video frame path.
    """
    frame_names = [
    p for p in os.listdir(frame_path)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    # print(frame_names)
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names

def load_image(frame_names, frame_idx, video_path="output_video_frames"):
    """
    Loads an image given frame names and index.
    """
    image = Image.open(os.path.join(video_path, frame_names[frame_idx]))
    return image

def plot_image(pil_img, labels, boxes):
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    
    # Create a mapping from labels to specific colors
    label_to_color = {label: COLORS[i % len(COLORS)] for i, label in enumerate(set(labels))}
    
    plt.figure(figsize=(12, 8))
    plt.imshow(pil_img)
    ax = plt.gca()
    
    for label, (xmin, ymin, xmax, ymax) in zip(labels, boxes):
        color = label_to_color[label]
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=color, linewidth=3))
        label_text = f'{label}'
        ax.text(xmin, ymin, label_text, fontsize=6, color='white')
    
    plt.axis('off')
    plt.show()

def dino_detect(processor, model, frame_names, start_frame, labels, box_thresholds, text_thresholds):
    """
    Returns detected objects of all labels in specified frame.
    """
    def dino_detect_single(processor, model, image, label, box_thresholds, text_thresholds):
        """
        Returns detected objects of a single label in an image.
        """
        # Process the image with the current label
        inputs = processor(images=image, text=[label], return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Postprocess results
        width, height = image.size
        postprocessed_outputs = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            target_sizes=[(height, width)],  # Single image
            box_threshold=box_thresholds,
            text_threshold=text_thresholds
        )
        return postprocessed_outputs
    
    image = load_image(frame_names, start_frame, video_path="output_video_frames")
    results = []
    for i, label in enumerate(labels):
        postprocessed_outputs = dino_detect_single(processor, model, image, label, box_thresholds[i], text_thresholds[i])
        # Store the result for the current label
        results.append(postprocessed_outputs[0])
    
    combined_scores, combined_labels, combined_boxes = combine_results(results)

    return combined_labels, combined_boxes

def yolo_detect(model, frame_names, start_frame, confidence, iou_threshold):
    """
    Returns detected objects of all labels in specified frame.
    """
    image = load_image(frame_names, start_frame, video_path="output_video_frames")
    results = model.infer(
        image=image,
        confidence=confidence,
        iou_threshold=iou_threshold
    )
    predictions = results[0].predictions
    bounding_boxes = [[pred.x, pred.y, pred.width, pred.height] for pred in predictions]
    converted_bboxes = [
        [
            x_center - width / 2,
            y_center - height / 2,
            x_center + width / 2,
            y_center + height / 2,
        ]
        for x_center, y_center, width, height in bounding_boxes
    ]
    labels = [pred.class_name for pred in predictions]
    return labels, converted_bboxes

def generate_bounding_box_dict(video_segments, frame_key):
    """
    Calculate bounding boxes for segmented objects from binary masks in a video frame.
    """
    annotations = []

    for obj_id, data in video_segments.items():
        mask = data["mask"].squeeze()
        label = data["label"]

        # Find the non-zero mask coordinates (where the object is segmented)
        coords = np.column_stack(np.where(mask > 0))  # coords is an array of [y, x] pairs

        if coords.shape[0] > 0:  # Check if the mask has any segmented pixels
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            width = xmax - xmin
            height = ymax - ymin

            # Calculate center coordinates
            x_center = xmin + width / 2.0
            y_center = ymin + height / 2.0

            # Append annotation in the specified format
            annotations.append({
                "id": obj_id,
                "label": label,
                "coordinates": {
                    "x": round(x_center, 2),
                    "y": round(y_center, 2),
                    "width": int(width),
                    "height": int(height)
                }
            })

    # Construct the dictionary for the frame
    frame_annotation = {
        "image": frame_key,
        "annotations": annotations
    }

    return frame_annotation

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_segmented_frames(video_path, frame_names, video_segments, start_frame, frame_stride, output_dir=None):
    """
    Returns list of frames with segmentation masks overlaid as PIL Images.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    frames_list = []  # List to store images

    for idx in tqdm(range(len(video_segments))):
        out_frame_idx = start_frame + idx*frame_stride  # Index into frame_names
        ann_frame_idx = extract_integer(frame_names[out_frame_idx])
        frame_path = os.path.join(video_path, frame_names[out_frame_idx])
        image = Image.open(frame_path)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(image)

        # Overlay masks
        for out_obj_id, data in video_segments[idx].items():
            out_mask = data["mask"]
            show_mask(out_mask, ax, obj_id=out_obj_id)

        # Save the figure to a buffer
        plt.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frame_image = Image.open(buf).convert('RGBA')
        frames_list.append(frame_image)
        buf.close()

        # Optionally save to file
        if output_dir is not None:
            output_file = os.path.join(output_dir, f"{ann_frame_idx}.png")
            frame_image.save(output_file)

    return frames_list

def save_gif(frames_list, output_gif_path, duration=100, loop=0):
    """
    Saves a list of PIL Image frames as an animated GIF.
    """
    if not frames_list:
        print("No frames to save.")
        return

    # Ensure all frames are in RGBA mode
    frames = [frame.convert('RGBA') for frame in frames_list]

    # Save the frames as an animated GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )

def save_frames_as_video(frames_list, output_video_path, fps=24):
    """
    Saves a list of PIL Image frames as an MP4 video using OpenCV.
    """
    if not frames_list:
        print("No frames to save.")
        return

    # Convert PIL Images to numpy arrays in BGR format
    frames = [cv2.cvtColor(np.array(frame.convert('RGB')), cv2.COLOR_RGB2BGR) for frame in frames_list]

    # Get frame dimensions
    height, width, layers = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")

# Main function
def process_video(video_path,
                  detector="Grounding DINO",
                  dino_box_threshold=[0.35, 0.35],
                  dino_text_threshold=[0.35, 0.35],
                  yolo_confidence=[0.4, 0.75],
                  yolo_iou_threshold=[0.8, 0.5],
                  frame_len=100,
                  frame_stride=5,
                  gif_duration=100):
    """
    """

    # Convert video to video frames
    print("Converting video to video frames")
    video_to_frames(video_path, output_dir="output_video_frames")

    # Select available device
    device = select_device()

    # Preprocess frames and labels
    frame_names = load_frames("output_video_frames")  # Frame name of all frames in directory
    width, height = load_image(frame_names, frame_idx=0).size

    if detector == "Grounding DINO":
        print("Object detection of starting frame using Grounding DINO")
        labels = ["player. referee", "basketball"]
        processed_labels = preprocess_caption(labels)
        print(processed_labels)
    
        # Load Grounding DINO
        processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")

        # Detect players, referee, and basketball in starting frame
        dino_frame_stride = frame_stride
        for idx in range(0, len(frame_names), dino_frame_stride):
            start_frame = idx
            detected_labels, detected_bboxes = dino_detect(processor, model, frame_names, 
                                                           start_frame, processed_labels,
                                                           dino_box_threshold, dino_text_threshold)
            string_counts = Counter(detected_labels)
            print(string_counts)
            if "basketball" in string_counts:
                if string_counts["basketball"] == 1 and string_counts["player"] >= 7:
                    print("Starting frame ", idx)
                    image = load_image(frame_names, start_frame, video_path="output_video_frames")
                    plot_image(image, detected_labels, detected_bboxes)
                    break

    elif detector == "YOLO":
        print("Object detection of starting frame using YOLO")
        load_dotenv()
        api_key_bball = os.getenv("ROBOFLOW_API_KEY_BASKETBALL")
        # api_key_all = os.getenv("ROBOFLOW_API_KEY_ALL")
        api_key_player_ref = os.getenv("ROBOFLOW_API_KEY_PLAYER_REFEREE")

        model_bball = inference.get_model(
            model_id="autozooming-basketball/2",
            api_key=api_key_bball
        )
        model_player_ref = inference.get_model(
            model_id="autozooming-players-referees/1",
            api_key=api_key_player_ref
        )
        # model_all = inference.get_model(
        #     model_id="basketball-game-object-detection-tdk5a/4",
        #     api_key=api_key_all
        # )
        yolo_frame_stride = 1
        for idx in range(0, len(frame_names), yolo_frame_stride):
            start_frame = idx
            # all_labels, all_bboxes = yolo_detect(model_all, frame_names, start_frame, 
            #                                     yolo_confidence[0], yolo_iou_threshold[0])
            player_ref_labels, player_ref_bboxes = yolo_detect(model_player_ref, frame_names, start_frame, 
                                                               yolo_confidence[0], yolo_iou_threshold[0])
            bball_labels, bball_bboxes = yolo_detect(model_bball, frame_names, start_frame, 
                                                    yolo_confidence[1], yolo_iou_threshold[1])
            string_counts = Counter(player_ref_labels)
            if len(bball_labels) == 1 and "player" in string_counts:
                player_count = string_counts["player"]
                if player_count >= 7:
                    print("Starting frame ", idx)
                    detected_labels = bball_labels + player_ref_labels
                    detected_bboxes = bball_bboxes + player_ref_bboxes
                    image = load_image(frame_names, start_frame, video_path="output_video_frames")
                    plot_image(image, detected_labels, detected_bboxes)
                    break
        
    else:
        raise ValueError("Select either 'Grounding DINO' or 'YOLO'")

    
    # Load sam2
    print("Loading SAM2")
    sam2_checkpoint = r"C:\Users\chris\sam2\checkpoints\sam2.1_hiera_large.pt"
    model_cfg = r"C:\Users\chris\sam2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # Get frames we want to track
    print("Initializing SAM2")
    frame_indices = []  # Frame number of all frames in directory
    for frame_name in frame_names:
        frame_indices.append(extract_integer(frame_name))
    
    filtered_frame_indices = [] # Frames we want to track
    counter = 0
    for idx in range(start_frame, len(frame_names), frame_stride):
        filtered_frame_indices.append(frame_indices[idx])
        counter += 1
        if counter >= frame_len:
            break

    inference_state = predictor.init_state(video_path=video_path, frame_indices=filtered_frame_indices)

    # Format the detected object bounding boxes and labels for sam2
    bounding_boxes = {}
    label_dict = {}
    id = 0
    for box, label in zip(detected_bboxes, detected_labels):
        bounding_boxes[id] = box
        label_dict[id] = label
        id += 1

    # Input annotation of starting frame for sam2
    for ann_obj_id in bounding_boxes:
        box = np.array(bounding_boxes[ann_obj_id], dtype=np.float32)
        ann_obj_id = int(ann_obj_id)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=ann_obj_id,
            box=box,
        )

    # Propagate segmentation throughout video
    video_segments = {}  # video_segments will store per-frame segmentation results with specific labels
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # Collect segmentation masks, applying the label from label_dict based on obj_id
        video_segments[out_frame_idx] = {
            out_obj_id: {
                "label": label_dict.get(out_obj_id, "unknown"),  # Fetch label from label_dict or default to "unknown"
                "mask": (out_mask_logits[i] > 0.0).cpu().numpy()
            }
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Create bounding box annotations for each frame
    print("Saving annotations")
    all_annot = []
    for idx in range(len(video_segments)):
        ann_frame_idx = extract_integer(frame_names[start_frame+idx*frame_stride])
        all_annot.append(generate_bounding_box_dict(video_segments[idx], f"{ann_frame_idx:05d}.jpg"))
    # print(all_annot)
    annot_file = r"output\annot.json"
    with open(annot_file, "w") as f:
        json.dump(all_annot, f, indent=4)

    # Save segmented frames and collect them in frames_list
    print("Saving segmented frames")
    frames_list = save_segmented_frames(
        video_path="output_video_frames",
        frame_names=frame_names,
        video_segments=video_segments,
        start_frame=start_frame,
        frame_stride=frame_stride,
        output_dir="output_segmented_frames"
    )

    # Create a GIF from the frames_list
    print("Creating video and gif")
    save_frames_as_video(frames_list, r"output\output.mp4", fps=24)
    save_gif(frames_list, r"output\output.gif", duration=gif_duration, loop=0)

    return
    
if __name__ == "__main__":
    video_path = r"Videos\Test Set 1.mp4"
    detector = "YOLO"
    dino_box_threshold = [0.35, 0.35]
    dino_text_threshold = [0.35, 0.35]
    yolo_confidence = [0.3, 0.75]
    yolo_iou_threshold = [0.7, 0.5]
    process_video(video_path=video_path,
                  detector=detector,
                  dino_box_threshold=dino_box_threshold,
                  dino_text_threshold=dino_text_threshold,
                  yolo_confidence=yolo_confidence,
                  yolo_iou_threshold=yolo_iou_threshold,
                  frame_len=50,
                  frame_stride=5,
                  gif_duration=200)