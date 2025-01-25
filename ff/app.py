from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

app = Flask(__name__)

# Path to the YOLO model
YOLO_MODEL_PATH = "yolov8n.pt"

# Step 1: Perform Object Detection with YOLO
def detect_objects(image_path):
    model = YOLO(YOLO_MODEL_PATH)  # Load the YOLOv8 nano model
    results = model(image_path)  # Perform inference on the image
    detected_objects = [model.names[int(box.cls)] for box in results[0].boxes]
    return detected_objects

# Step 2: Create a Kid-Friendly Prompt
def create_kid_friendly_prompt(detected_objects):
    prompt = "On a sunny day, "
    if detected_objects:
        prompt += f"I saw {', '.join(detected_objects)}. "
    else:
        prompt += "I saw nothing unusual. "
    prompt += "Then something amazing happened..."
    return prompt

# Step 3: Generate a Story Using GPT-2
def generate_story(prompt):
    model_name = "gpt2"  # Pre-trained GPT-2 model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    inputs = tokenizer.encode(prompt, return_tensors="pt",min_length=200, max_length=500, truncation=True)

    outputs = model.generate(
        inputs,
        min_length=200,
        max_length=500,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2
    )

    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

# Route to handle file upload and story generation
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        image = request.files['image']
        if image.filename == '':
            return redirect(request.url)
        if image:
            # Save the image to static/images
            image_path = os.path.join('static/images',image.filename )
            image.save(image_path)

            # Perform object detection
            detected_objects = detect_objects(image_path)

            # Create a kid-friendly prompt
            prompt = create_kid_friendly_prompt(detected_objects)

            # Generate a story
            story = generate_story(prompt)

            return render_template('index.html', story=story, detected_objects=detected_objects)

    return render_template('index.html', story=None, detected_objects=None)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image uploaded", 400
        image = request.files['image']
        if image.filename == '':
            return "No image selected", 400
        if image:
            # Save the image
            image_path = os.path.join('static/uploads', image.filename)
            image.save(image_path)

            # Process the image
            detected_objects = detect_objects(image_path)
            story = generate_story(detected_objects)

            return render_template('index.html', story=story, detected_objects=detected_objects)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)











