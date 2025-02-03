
# for recognition
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests

# for generation
from diffusers import StableDiffusionPipeline
import torch
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4
torch.cuda.empty_cache()


model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

safety_checker = pipe.safety_checker
feature_extractor = pipe.feature_extractor

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_type=torch.float16, revision="fp16", safety_checker = None)



# negative_prompt = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"



# -- Input --

url = input("Enter your url for starting image: ")

loops = int(input("Enter amount of loops: "))

likeness = float(input("Enter threshold for image recognition (percentage): "))

# -- Starting recognition --

image = Image.open(requests.get(url, stream=True).raw)

# get image dimensions
width = image.width
height = image.height

# you can specify the revision tag if you don't want the timm dependency
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
promptStr = " ".join(str(item) for item in prompt)

# -- Ouroboros looping --

i = 1
while(i < (loops + 1)):

    # generator
    image = pipe(promptStr).images[0]
    image.save("Ouro_" + str(i) + "_.png")

    # recognizer
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    labels = results["labels"]
    prompt = []
    for l in labels:
        prompt.append(model.config.id2label[l.item()])

    promptStr = " ".join(str(item) for item in prompt)
    # incr
    i += 1

