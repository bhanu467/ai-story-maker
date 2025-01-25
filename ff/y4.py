from ultralytics import YOLO
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Perform Object Detection with YOLO
def detect_objects(image_path):
    """
    Detect objects in the image using YOLOv8.
    """
    model = YOLO("yolov8n.pt")  # Load the YOLOv8 nano model
    results = model(image_path)  # Perform inference on the image

    # Extract detected object names
    detected_objects = [model.names[int(box.cls)] for box in results[0].boxes]
    return detected_objects

# Step 2: Create a Kid-Friendly Prompt
def create_kid_friendly_prompt(detected_objects):
    """
    Create a simple, child-friendly story prompt based on detected objects.
    """
    prompt = "On a sunny day, "
    if detected_objects:
        prompt += f"I saw {', '.join(detected_objects)}. "
    else:
        prompt += "I saw nothing unusual. "
    prompt += "Then something amazing happened..."
    return prompt

# Step 3: Generate a Story Using Hugging Face GPT-2
def generate_story(prompt):
    """
    Generate a kid-friendly story based on the input prompt using GPT-2.
    """
    model_name = "gpt2"  # Pre-trained GPT-2 model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=100, truncation=True)

    # Generate story
    outputs = model.generate(
        inputs,
        max_length=200,  # Set a limit for story length
        num_return_sequences=1,  # Generate one story
        temperature=0.7,  # Adjust creativity
        top_p=0.9,  # Nucleus sampling
        do_sample=True,  # Enable sampling for diversity
        repetition_penalty=1.2  # Avoid repetitive text
    )

    # Decode the generated story
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

# Step 4: Connect Object Detection and Story Generation
def main(image_path):
    """
    Main function to perform object detection and generate a kid-friendly story.
    """
    # Perform object detection
    detected_objects = detect_objects(image_path)
    print("Detected Objects:", detected_objects)

    # Create a prompt for the story
    prompt = create_kid_friendly_prompt(detected_objects)
    print("Story Prompt:", prompt)

    # Generate a story
    story = generate_story(prompt)
    print("\nGenerated Story:\n", story)

# Run the program
if __name__ == "__main__":
    image_path = "cat.jpeg"  # Replace with the path to your image
    main(image_path)
