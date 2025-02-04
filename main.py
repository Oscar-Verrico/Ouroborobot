
# for recognition
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageTk
import requests

# for generation
from diffusers import StableDiffusionPipeline
import torch
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4
torch.cuda.empty_cache()

# for ui
import threading
import tkinter as tk
import os
from tkinter import filedialog, messagebox, StringVar, IntVar, DoubleVar


# models
recognition_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
recognition_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

# ui setup
root = tk.Tk()
root.title("Ouroborobot.exe")
root.geometry("800x600")
root.resizable(False,False)

# variables
selected_image_path = StringVar()
num_loops = IntVar(value=1)
negative_prompt = StringVar()
output_directory = StringVar()
likeness_threshold = DoubleVar(value=95.0)  # Default to 95% likeness
log_messages = tk.StringVar()
obj_list = tk.StringVar()








# HELPER FUNCTIONS

# Selects the pipe mode (Cuda or CPU)
def pipe_mode(pipe_obj):
    if torch.cuda.is_available():
        pipe_obj.to("cuda")
        log_message("Using CUDA for pipeline.")
    else:
        pipe_obj.to("cpu")
        log_message("Using CPU for pipeline.")

# Selects starting image
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        selected_image_path.set(file_path)
        display_selected_image(file_path)

# Loads image in UI
def display_selected_image(file_path):
    image = Image.open(file_path).resize((200,200))
    image_tk = ImageTk.PhotoImage(image)
    image_label.config(image=image_tk)
    image_label.image = image_tk

# Selects output files location
def select_output_directory():
    directory = filedialog.askdirectory()
    if directory:
        output_directory.set(directory)

# Adds messages to the log of the program
def log_message(message):
    current_log = log_messages.get()
    log_messages.set(current_log + f"\n{message}")
    log_textbox.config(state=tk.NORMAL)
    log_textbox.insert(tk.END, f"{message}\n")
    log_textbox.config(state=tk.DISABLED)
    log_textbox.yview(tk.END)

# runs the main process pipeline
def run_pipeline():
    pipe_mode(pipe)
    if not selected_image_path.get() or not output_directory.get():
        messagebox.showerror("Error", "Please select both a starting image and an output directory.")
        return

    log_message("Starting image recognition and generation process...")

    # Run the process in a separate thread to avoid freezing the GUI
    thread = threading.Thread(target=pipeline_thread)
    thread.start()

# Threads process to be in background 
def pipeline_thread():
    prompt_str = process_image(selected_image_path.get(), likeness_threshold.get() / 100.0)
    iterations = num_loops.get()

    for i in range(1, iterations + 1):
        log_message(f"Iteration {i}/{iterations}: Generating image...")
        new_image = generate_image(prompt_str)
        output_path = os.path.join(output_directory.get(), f"Ouro_{i}.png")
        new_image.save(output_path)
        log_message(f"Image saved: {output_path}")

        # Process the generated image and update the prompt
        prompt_str = process_image(output_path, likeness_threshold.get() / 100.0)
        display_selected_image(output_path)

    log_message("Process completed.")

# Recognize objects in the given image and return a prompt string.
def process_image(image_path, threshold):
    image = Image.open(image_path).convert("RGB")
    inputs = recognition_processor(images=image, return_tensors="pt")
    outputs = recognition_model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = recognition_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    detected_objects = [recognition_model.config.id2label[label.item()] for label in results["labels"]]
    obj_list.set("\n".join(detected_objects))
    log_message(f"Detected objects: {detected_objects}")
    
    return " ".join(detected_objects)

# Generate an image using the given prompt and Stable Diffusion.
def generate_image(prompt):
    negative_prompt_text = negative_prompt.get()
    if negative_prompt_text:
        log_message(f"Applying negative prompt: {negative_prompt_text}")

    return pipe(prompt).images[0]









# GUI Layout

# Image selection frame
image_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
image_frame.place(x=20, y=20, width=220, height=220)
image_label = tk.Label(image_frame, text="No image selected", padx=10, pady=10)
image_label.pack()

# Buttons and text input fields
tk.Button(root, text="Select Starting Image", command=select_image).place(x=20, y=250)
tk.Button(root, text="Select Output Folder", command=select_output_directory).place(x=20, y=290)
tk.Label(root, text="Number of Iterations:").place(x=20, y=330)
tk.Entry(root, textvariable=num_loops, width=5).place(x=150, y=330)
tk.Label(root, text="Negative Prompt:").place(x=20, y=360)
tk.Entry(root, textvariable=negative_prompt, width=50).place(x=150, y=360)
tk.Label(root, text="Likeness Threshold (%):").place(x=20, y=390)
tk.Entry(root, textvariable=likeness_threshold, width=5).place(x=150, y=390)

# Detected objects list
tk.Label(root, text="Detected Objects:").place(x=300, y=20)
tk.Label(root, textvariable=obj_list, justify="left", anchor="nw", width=30, height=10, relief=tk.SUNKEN).place(x=300, y=50)

# Output directory
tk.Label(root, text="Output Directory:").place(x=300, y=250)
tk.Entry(root, textvariable=output_directory, width=40).place(x=400, y=250)

# Loop counter (displayed but does not need user interaction)
tk.Label(root, text="Loop Counter:").place(x=300, y=290)
loop_counter_label = tk.Label(root, textvariable=num_loops, relief=tk.SUNKEN, width=5)
loop_counter_label.place(x=400, y=290)

# Log section
tk.Label(root, text="Log:").place(x=20, y=430)
log_textbox = tk.Text(root, height=10, width=90, state=tk.DISABLED)
log_textbox.place(x=20, y=460)

# Start button
tk.Button(root, text="Start Process", command=run_pipeline, bg="green", fg="white").place(x=350, y=550)

root.mainloop()
