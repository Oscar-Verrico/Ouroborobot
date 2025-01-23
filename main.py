
# for recognition
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests

from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")






# -- Input --

url = input("Enter your url for starting image: ")

loops = int(input("Enter amount of loops: "))

likeness = float(input("Enter threshold for image recognition (percentage): "))

# -- Starting recognition --

image = Image.open(requests.get(url, stream=True).raw)

# get image dimensions
width = image.width
height = image.height

# model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=(likeness/100.0))[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )
labels = results["labels"]
prompt = []
num_of_objs = 0

for l in labels:
    prompt.append(model.config.id2label[l.item()])
    num_of_objs = num_of_objs + 1

print("Detected objects in starter image: ")
print(prompt)


# -- Ouroboros looping --

i = 1
while(i < (loops + 1)):

    # generator
    image = pipe(prompt).images[0]
    image.save("astronaut_rides_horse_" + i + "_.png")

    # recognizer
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    labels = results["labels"]
    prompt = []
    for l in labels:
        prompt.append(model.config.id2label[l.item()])

    # incr
    i += 1

