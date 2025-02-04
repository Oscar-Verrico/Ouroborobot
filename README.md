The **Ouroborobot** is a Python-based GUI program that iteravely performs a looping of 
object recognition and image generation using AI models. It functions as a game of telephone with a 
starting image provided, then analyzed to note objects detected, then generates an image with that list of objects,
and so on and so forth for as long as desired, and have extra options with likeliness threshold changing for 
object detection and negative prompts to prevent certain objects from generating.

Key features include:

* Object detection using DETR (Detection Transformer).
* Image generation using Stable Diffusion.
* Real-time updates in the GUI, displaying detected objects and log progress.
* Customizable settings for iterations, likeness threshold, and negative prompts.

To set up the program and install dependencies, run this in a Python 3.10 environment:

pip install -r requirements.txt

This will install all the necessary libraries, including PyTorch, Transformers, and Diffusers, 
which are used for object detection and image generation.
