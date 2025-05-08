BASKETBALL GAME AUTOZOOM

This basketball game autozoom program can automatically perform zooming of a basketball game. To run this, you will need to install the following libraries (see installation instruction links):

Pytorch: https://pytorch.org/get-started/locally/

Inference from Roboflow: https://inference.roboflow.com/quickstart/run_a_model/

Hugging Face transformers: https://huggingface.co/docs/transformers/en/installation

SAM2 from Meta: https://github.com/facebookresearch/sam2

HMR2.0 requirement: https://github.com/shubham-goel/4D-Humans


If running locally, you will need to change the SAM2 reference directory in the obj_detect.py script to reference the installation location of SAM2.

To run the entire pipeline, you may run the entire gradio_bball_zoom.ipynb notebook and use the linked gradio interface to upload your basketball game video for autozooming.


Below are the explanations for the other parameters:

Grounding DINO or YOLO - Object detector to use as well as the detection parameters. 

Frame length (how many frames you want in your autozoomed video)

Frame stride (how many frames to skip in the pipeline)

GIF duration (how long each frame of the GIF pauses for)

"Use previous data" if you've already generated annotated images from a previous run and just want to play around with the autozoom settings.

"Use pose data" to include pose data for autozooming. 

"Out filename" is the filename of the autozoomed output GIF. 

Below are the explanations for the other parameters:

FPS - Frames per second of the output gif 

IDs Targeted (separated by commas) - The IDs of targeted objects (as written in the annotation JSON file). Used for debugging. 

Labels Targeted (separated by commas) - The labels of things to target ex: "basketball, player, referee" 

Closest to Ball - Slider of how many players closest to the ball should be tracked. 

X Padding - Padding in the x-direction away from detected tracked objects 

Y Padding - Padding in the y-direction away from detected tracked objects
