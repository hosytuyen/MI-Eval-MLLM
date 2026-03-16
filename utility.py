import os
import random
import re
import base64
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import google.generativeai as genai
from PIL import Image
import csv
import pandas as pd
try:
    import openai
except ImportError:
    openai = None
# Your crop function (for CelebA)
def crop(x):
    return x[:, 55:55 + 108, 35:35 + 108]

# Create processor that applies crop to PIL image
processor = transforms.Compose([
    transforms.ToTensor(),        # PIL → Tensor
    # transforms.Lambda(crop),      # Crop the tensor
    transforms.ToPILImage(),      # Tensor → PIL
    transforms.Resize((224, 224))   # Resize final output
])

def create_concatenated_images_with_labels(private_image_folder, mi_folder, output_folder, num_set_b_images=4, model_name=None):
    """
    Create concatenated images with labels for MI evaluation.
    
    Args:
        private_image_folder: Path to private data folder
        mi_folder: Path to MI attack images folder
        output_folder: Path to output folder for concatenated images
        num_set_b_images: Number of images in Set B (default: 4)
        model_name: Model name (e.g., 'gpt-5') to determine special prompts (default: None)
    """
    # Define white space width and text space height
    white_space_width = 100
    text_space_height = 50  # Space for labels
    if model_name == "gpt-5":
        question_space_height = 80
    else:
        question_space_height = 50
    image_size = 224  # Size of each image

    # Scale font sizes based on number of images in Set B
    # Smaller fonts for fewer images to fit better
    if num_set_b_images <= 2:
        label_font_size = 20
        question_font_size = 24
    elif num_set_b_images <= 4:
        label_font_size = 30
        question_font_size = 35
    else:  # 5 or more images
        label_font_size = 30
        question_font_size = 35

    # Define font
    try:
        font = ImageFont.truetype("arial.ttf", label_font_size)
        question_font = ImageFont.truetype("arial.ttf", question_font_size)
    except IOError:
        # Default font if arial is not available
        font = ImageFont.load_default()
        question_font = ImageFont.load_default()

    # Create the output folder with the same structure as mi_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Go through each sub-folder in mi_folder
    for subfolder_name in os.listdir(mi_folder):
        # Define paths to the corresponding subfolders
        private_subfolder = os.path.join(private_image_folder, subfolder_name)
        mi_subfolder = os.path.join(mi_folder, subfolder_name)
        output_subfolder = os.path.join(output_folder, subfolder_name)
        
        # Ensure subfolder exists in private_image_folder
        if not os.path.isdir(private_subfolder):
            print(f"Skipping {subfolder_name} as it is not in private_image_folder.")
            continue

        # Create the output subfolder
        Path(output_subfolder).mkdir(parents=True, exist_ok=True)
        
        # Get list of images in each subfolder
        private_images = [os.path.join(private_subfolder, f) for f in os.listdir(private_subfolder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        mi_images = [os.path.join(mi_subfolder, f) for f in os.listdir(mi_subfolder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Check if we have enough private images
        if len(private_images) < num_set_b_images:
            print(f"Warning: Subfolder {subfolder_name} has only {len(private_images)} private images, but {num_set_b_images} requested. Using all available.")
        
        # Process each image in mi_folder
        for mi_image_path in mi_images:
            # Open the mi image and resize to 224x224
            mi_image = Image.open(mi_image_path).resize((image_size, image_size))

            # Select num_set_b_images random images from the private images
            if len(private_images) >= num_set_b_images:
                selected_private_images = random.sample(private_images, num_set_b_images)
            else:
                # If not enough images, use all available and pad with duplicates
                selected_private_images = list(private_images)
                while len(selected_private_images) < num_set_b_images:
                    selected_private_images.append(random.choice(private_images))
            
            private_images_resized = []
            for img in selected_private_images:
                image = Image.open(img)
                if any(attack in mi_folder for attack in ["KEDMI", "LOMMA", "PLGMI"]):
                    image = processor(image)
                else:
                    image = image.resize((image_size, image_size))
                private_images_resized.append(image)
                
            # Calculate the width and height for the new concatenated image
            # Image A (224) + white space (100) + Set B images (224 * num_set_b_images)
            total_width = image_size + white_space_width + (image_size * num_set_b_images)
            total_height = image_size + text_space_height + question_space_height
            concatenated_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))  # White background
            
            # Paste the mi image (Image A)
            concatenated_image.paste(mi_image, (0, 0))
            
            # Paste Set B images with white space after Image A
            x_offset = image_size + white_space_width
            for private_image in private_images_resized:
                concatenated_image.paste(private_image, (x_offset, 0))
                x_offset += image_size
            
            # Draw text labels and question
            draw = ImageDraw.Draw(concatenated_image)
            
            # Draw "Image A" label (centered below Image A)
            image_a_center_x = image_size / 2
            draw.text((image_a_center_x, image_size + 5 + 20), "Image A", fill="black", font=font, anchor="mm")
            
            # Draw "Set of Images B" label (centered over Set B images)
            set_b_start_x = image_size + white_space_width
            set_b_width = image_size * num_set_b_images
            set_b_center_x = set_b_start_x + (set_b_width / 2)
            draw.text((set_b_center_x, image_size + 5 + 20), "Set of Images B", fill="black", font=font, anchor="mm")
            
            # Detect dataset type and adapt question
            path_lower = private_image_folder.lower()
            if "cifar" in path_lower or "cifa" in path_lower:
                question = "Does Image A depict the same object class as the images in Set B?"
            elif "facescrub" in path_lower or "celeb" in path_lower:
                question = "Does Image A depict the same individual as the images in Set B?"
                if model_name == "gpt-5":
                    question = "Based on only facial features and ignoring identity, \n are the faces in Image A and Set B visually similar?"
            else:
                question = "Does Image A depict the same breed as the images in Set B?"
            
            # Draw question in red (centered over entire image)
            draw.text((total_width / 2, image_size + text_space_height + 5 + 20), 
                      question, 
                      fill="red", font=question_font, anchor="mm")

            # Define the output image path with the same name as the mi image
            output_image_path = os.path.join(output_subfolder, os.path.basename(mi_image_path))
            concatenated_image.save(output_image_path)


def create_positive_pair_user_study_images(private_image_folder, output_folder, images_per_class=10):
    """
    Create positive pair user study images where:
    - Image A: random image from class x
    - Set B: 4 random images from class x (same class)
    
    Args:
        private_image_folder: Path to private data folder with class subfolders (0, 1, 2, ...)
        output_folder: Path to output folder for positive pair images
        images_per_class: Number of positive pair images to create per class
    """
    # Detect dataset type from path
    path_lower = private_image_folder.lower()
    if "cifar" in path_lower or "cifa" in path_lower:
        question = "Does Image A depict the same object class as the images in Set B?"
    elif "facescrub" in path_lower or "celeb" in path_lower:
        question = "Does Image A depict the same individual as the images in Set B?"
        # question = "Based on only facial features and ignoring identity, \n are the faces in Image A and Set B visually similar?"
    else:
        question = "Does Image A depict the same breed as the images in Set B?"
    # Define white space width and text space height
    white_space_width = 100
    text_space_height = 50  # Space for labels
    question_space_height = 50  # Space for question text
    # question_space_height = 80  # Space for question text

    # Define font
    try:
        font = ImageFont.truetype("arial.ttf", 30)
        question_font = ImageFont.truetype("arial.ttf", 35)
    except IOError:
        # Default font if arial is not available
        font = ImageFont.load_default()
        question_font = ImageFont.load_default()

    # Create the output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all class folders
    class_folders = sorted([d for d in os.listdir(private_image_folder) 
                           if os.path.isdir(os.path.join(private_image_folder, d)) and d.isdigit()])
    
    print(f"Creating positive pair images for {len(class_folders)} classes...")
    
    # Process each class folder
    for class_id in class_folders:
        class_folder = os.path.join(private_image_folder, class_id)
        output_class_folder = os.path.join(output_folder, class_id)
        Path(output_class_folder).mkdir(parents=True, exist_ok=True)
        
        # Get all images in this class
        images = [os.path.join(class_folder, f) for f in os.listdir(class_folder) 
                if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) < 5:
            print(f"Skipping class {class_id}: not enough images (need at least 5, have {len(images)})")
            continue
        
        # Create images_per_class positive pair images
        for i in range(images_per_class):
            # Select Image A (random image from class x)
            image_a_path = random.choice(images)
            image_a = Image.open(image_a_path).resize((224, 224))
            
            # Select 4 random images from the same class (excluding Image A)
            remaining_images = [img for img in images if img != image_a_path]
            if len(remaining_images) < 4:
                # If not enough remaining images, allow duplicates
                selected_images = random.choices(remaining_images, k=4)
            else:
                selected_images = random.sample(remaining_images, 4)
            
            # Resize selected images
            set_b_images = [Image.open(img).resize((224, 224)) for img in selected_images]
            
            # Calculate the width and height for the new concatenated image
            total_width = (224 * 5) + white_space_width
            total_height = 224 + text_space_height + question_space_height
            concatenated_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))  # White background
            
            # Paste Image A and Set B images with white space in between
            concatenated_image.paste(image_a, (0, 0))
            x_offset = 224 + white_space_width
            for img in set_b_images:
                concatenated_image.paste(img, (x_offset, 0))
                x_offset += 224
            
            # Draw text labels and question
            draw = ImageDraw.Draw(concatenated_image)
            # Draw "Image A" label
            draw.text((112, 224 + 5 + 20), "Image A", fill="black", font=font, anchor="mm")
            
            # Draw "Set of Images B" label
            draw.text((224 * 3 + white_space_width / 2, 224 + 5 + 20), "Set of Images B", fill="black", font=font, anchor="mm")
            
            # Draw question in red (adapted for dataset type)
            draw.text((total_width / 2, 224 + text_space_height + 5 + 20), 
                      question, 
                      fill="red", font=question_font, anchor="mm")

            # Define the output image path
            image_a_name = os.path.basename(image_a_path)
            base_name = os.path.splitext(image_a_name)[0]
            output_image_path = os.path.join(output_class_folder, f"positive_{base_name}_{i}.png")
            concatenated_image.save(output_image_path)
    
    print(f"Positive pair images saved to {output_folder}")


def create_negative_pair_user_study_images(private_image_folder, output_folder, images_per_class=10):
    """
    Create negative pair user study images where:
    - Image A: random image from class x
    - Set B: 4 random images from class y (different class, randomly chosen)
    
    Args:
        private_image_folder: Path to private data folder with class subfolders (0, 1, 2, ...)
        output_folder: Path to output folder for negative pair images
        images_per_class: Number of negative pair images to create per class
    """
    # Detect dataset type from path
    path_lower = private_image_folder.lower()
    if "cifar" in path_lower or "cifa" in path_lower:
        question = "Does Image A depict the same object class as the images in Set B?"
    elif "facescrub" in path_lower or "celeb" in path_lower:
        question = "Does Image A depict the same individual as the images in Set B?"
        # question = "Based on only facial features and ignoring identity, \n are the faces in Image A and Set B visually similar?"
    else:
        question = "Does Image A depict the same breed as the images in Set B?"
    # Define white space width and text space height
    white_space_width = 100
    text_space_height = 50  # Space for labels
    question_space_height = 50  # Space for question text
    # Define font
    try:
        font = ImageFont.truetype("arial.ttf", 30)
        question_font = ImageFont.truetype("arial.ttf", 35)
    except IOError:
        # Default font if arial is not available
        font = ImageFont.load_default()
        question_font = ImageFont.load_default()

    # Create the output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all class folders
    class_folders = sorted([d for d in os.listdir(private_image_folder) 
                           if os.path.isdir(os.path.join(private_image_folder, d)) and d.isdigit()])
    
    print(f"Creating negative pair images for {len(class_folders)} classes...")
    
    # Process each class folder
    for class_id in class_folders:
        class_folder = os.path.join(private_image_folder, class_id)
        output_class_folder = os.path.join(output_folder, class_id)
        Path(output_class_folder).mkdir(parents=True, exist_ok=True)
        
        # Get all images in this class (for Image A)
        images_a = [os.path.join(class_folder, f) for f in os.listdir(class_folder) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images_a) == 0:
            print(f"Skipping class {class_id}: no images found")
            continue
        
        # Get other class folders (for Set B)
        other_classes = [c for c in class_folders if c != class_id]
        
        if len(other_classes) == 0:
            print(f"Skipping class {class_id}: no other classes available")
            continue
        
        # Create images_per_class negative pair images
        for i in range(images_per_class):
            # Select Image A (random image from class x)
            image_a_path = random.choice(images_a)
            image_a = Image.open(image_a_path).resize((224, 224))
            
            # Randomly choose a different class for Set B
            other_class_id = random.choice(other_classes)
            other_class_folder = os.path.join(private_image_folder, other_class_id)
            
            # Get images from the other class
            images_b = [os.path.join(other_class_folder, f) for f in os.listdir(other_class_folder) 
                       if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(images_b) < 4:
                print(f"Warning: Class {other_class_id} has only {len(images_b)} images, using all available")
                selected_images = images_b if len(images_b) > 0 else []
                # If still not enough, pad with duplicates
                while len(selected_images) < 4 and len(images_b) > 0:
                    selected_images.append(random.choice(images_b))
            else:
                selected_images = random.sample(images_b, 4)
            
            # Resize selected images
            set_b_images = [Image.open(img).resize((224, 224)) for img in selected_images]
            
            # Calculate the width and height for the new concatenated image
            total_width = (224 * 5) + white_space_width
            total_height = 224 + text_space_height + question_space_height
            concatenated_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))  # White background
            
            # Paste Image A and Set B images with white space in between
            concatenated_image.paste(image_a, (0, 0))
            x_offset = 224 + white_space_width
            for img in set_b_images:
                concatenated_image.paste(img, (x_offset, 0))
                x_offset += 224
            
            # Draw text labels and question
            draw = ImageDraw.Draw(concatenated_image)
            # Draw "Image A" label
            draw.text((112, 224 + 5 + 20), "Image A", fill="black", font=font, anchor="mm")
            
            # Draw "Set of Images B" label
            draw.text((224 * 3 + white_space_width / 2, 224 + 5 + 20), "Set of Images B", fill="black", font=font, anchor="mm")
            
            # Draw question in red (adapted for dataset type)
            draw.text((total_width / 2, 224 + text_space_height + 5 + 20), 
                      question, 
                      fill="red", font=question_font, anchor="mm")

            # Define the output image path
            image_a_name = os.path.basename(image_a_path)
            base_name = os.path.splitext(image_a_name)[0]
            output_image_path = os.path.join(output_class_folder, f"negative_{base_name}_{i}.png")
            concatenated_image.save(output_image_path)
    
    print(f"Negative pair images saved to {output_folder}")


def normalize_response(response):
    """Normalize the response to handle different cases of 'yes' and 'no'"""
    if response is None:
        return False
    response = response.strip().lower()
    # Remove trailing punctuation
    response = response.rstrip('.,!?;:')
    # Check for yes variations
    if response == 'yes':
        return True
    # Check for no variations
    if response == 'no':
        return False
    # If response contains 'yes' as a word, return True
    import re
    if re.search(r'\byes\b', response):
        return True
    # If response contains 'no' as a word (but not 'not', 'nothing', 'now', 'know'), return False
    if re.search(r'\bno\b', response):
        # Make sure it's not part of other words
        if not re.search(r'\bnot\b', response) and \
           not re.search(r'\bnothing\b', response) and \
           not re.search(r'\bnow\b', response) and \
           not re.search(r'\bknow\b', response) and \
           not re.search(r'\bknown\b', response):
            return False
    # Default to False if unclear
    return False

def is_refusal(response):
    """Check if the response indicates refusal to answer"""
    if response is None:
        return True
    response_lower = response.strip().lower()
    
    # Common refusal phrases (more specific to avoid false positives)
    refusal_phrases = [
        'cannot', 'can\'t', 'unable to', 'refuse', 'decline', 
        'not able to', 'not allowed', 'not permitted', 'not appropriate',
        'i cannot', 'i can\'t', 'i\'m unable', 'i refuse',
        'i don\'t', 'i do not', 'i won\'t', 'i will not',
        'sorry,', 'i apologize', 'not comfortable', 'not ethical',
        'cannot determine', 'cannot identify', 'cannot recognize', 
        'cannot answer', 'unable to determine', 'unable to identify',
        'unable to recognize', 'unable to answer', 'i\'m not able',
        'i cannot determine', 'i cannot identify', 'i cannot recognize',
        'i cannot answer', 'i\'m not comfortable', 'i decline',
        'i refuse to', 'i will not answer', 'i won\'t answer'
    ]
    
    # Check if response contains refusal phrases
    for phrase in refusal_phrases:
        if phrase in response_lower:
            return True
    
    # Check if response is very short and doesn't contain yes/no
    if len(response_lower) < 3:
        return True
    
    # Check if response doesn't contain yes or no at all (might be a refusal)
    if 'yes' not in response_lower and 'no' not in response_lower:
        # But allow if it's a clear affirmative/negative word
        affirmative_words = ['correct', 'true', 'same', 'match', 'identical']
        negative_words = ['incorrect', 'false', 'different', 'not the same', 'not match']
        if not any(word in response_lower for word in affirmative_words + negative_words):
            return True
    
    return False

def categorize_response(response):
    """
    Categorize Gemini response into: 'yes', 'no', or 'refuse'
    Handles variations like "yes, Yes, yes." or "no, No, no."
    
    First checks if we can get exactly "yes" or "no" (after considering variations),
    otherwise checks if it's a refusal.
    
    Returns:
        str: 'yes', 'no', or 'refuse'
    """
    if response is None:
        return 'refuse'
    
    response_lower = response.strip().lower()
    
    # Normalize the response: remove punctuation and extra whitespace for matching
    # But keep original for word boundary checks
    response_normalized = re.sub(r'[^\w\s]', ' ', response_lower)
    response_normalized = ' '.join(response_normalized.split())
    
    # First, check for yes - handle variations like "yes", "yes,", "yes.", "Yes", "yes, Yes, yes."
    # Look for "yes" as a standalone word or at word boundaries
    if re.search(r'\byes\b', response_normalized):
        return 'yes'
    
    # Also check if response starts or ends with "yes" followed by punctuation
    if re.match(r'^yes[\s,\.!?]', response_lower) or re.search(r'[\s,\.!?]yes[\s,\.!?]*$', response_lower):
        return 'yes'
    
    # Second, check for no - handle variations like "no", "no,", "no.", "No", "no, No, no."
    # But be careful not to match "not", "nothing", "now", "know", etc.
    # First check if "no" appears as a standalone word
    if re.search(r'\bno\b', response_normalized):
        # Make sure it's not part of "not", "nothing", "now", "know", "known"
        if not re.search(r'\bnot\b', response_normalized) and \
           not re.search(r'\bnothing\b', response_normalized) and \
           not re.search(r'\bnow\b', response_normalized) and \
           not re.search(r'\bknow\b', response_normalized) and \
           not re.search(r'\bknown\b', response_normalized):
            return 'no'
    
    # Also check if response starts or ends with "no" followed by punctuation
    if re.match(r'^no[\s,\.!?]', response_lower) or re.search(r'[\s,\.!?]no[\s,\.!?]*$', response_lower):
        # Additional check to avoid false positives
        words = response_normalized.split()
        if 'no' in words and 'not' not in words and 'nothing' not in words:
            return 'no'
    
    # If we can't get exactly yes or no, check if it's a refusal
    if is_refusal(response_lower):
        return 'refuse'
    
    # If unclear and not a clear refusal, treat as refusal
    return 'refuse'

# ===== Unified Evaluation Framework =====

def get_dataset_type_from_path(path):
    """Detect dataset type from path"""
    path_lower = path.lower()
    if "cifar" in path_lower or "cifa" in path_lower:
        return "object"
    elif "facescrub" in path_lower or "celeb" in path_lower:
        return "face"
    else:
        return "dog"

def get_prompt_for_image(image_path, task_type="regular"):
    """
    Generate prompt based on dataset type and task type.
    
    Args:
        image_path: Path to image
        task_type: "regular" or "user_study"
    
    Returns:
        str: Generated prompt
    """
    dataset_type = get_dataset_type_from_path(image_path)
    
    if task_type == "regular":
        prompts = {
            "object": "You are an expert in CIFAR100 object recognition. Taking into account different lighting, angles, backgrounds, and variations in appearance, do the task in the image. Only answer yes or no",
            "face": "You are an expert in face recognition. Taking into account the face aging, lighting, different hair styles, wearing and not wearing of eye glasses or other accessory, do the task in the image. Only answer yes or no",
            "dog": "You are an expert in dog breed recognition. Taking into account different lighting, angles, backgrounds, and variations in appearance, do the task in the image. Only answer yes or no"
        }
    else:  # user_study
        prompts = {
            "object": "You are an expert in CIFAR100 object recognition. Look at the image carefully. Does Image A depict the same object class as the images in Set B? Only answer yes or no. If you cannot determine or are not comfortable answering, you may refuse to answer.",
            "face": "You are an expert in face recognition. Look at the image carefully. Does Image A depict the same individual as the images in Set B? Only answer yes or no. If you cannot determine or are not comfortable answering, you may refuse to answer.",
            "dog": "You are an expert in dog breed recognition. Look at the image carefully. Does Image A depict the same breed as the images in Set B? Only answer yes or no. If you cannot determine or are not comfortable answering, you may refuse to answer."
        }
    
    return prompts[dataset_type]

class ModelProcessor:
    """Base class for model processors"""
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def process_image(self, image_path, prompt):
        """Process a single image with the model. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def initialize(self):
        """Initialize model-specific resources"""
        pass

class GeminiProcessor(ModelProcessor):
    def __init__(self, api_key, model_name):
        super().__init__(api_key=api_key, model_name=model_name)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def process_image(self, image_path, prompt):
        try:
            image = Image.open(image_path)
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

class QwenVLProcessor(ModelProcessor):
    def __init__(self, server_url, model_name, api_name="/process_query"):
        super().__init__(server_url=server_url, model_name=model_name, api_name=api_name)
        from gradio_client import Client
        self.client = Client(server_url)
        self.api_name = api_name
    
    def process_image(self, image_path, prompt):
        try:
            from gradio_client import handle_file
            result = self.client.predict(
                text_prompt=prompt,
                image=handle_file(image_path),
                max_tokens=512,
                temperature=2,
                top_p=1,
                top_k=100,
                repetition_penalty=2,
                do_sample=True,
                api_name=self.api_name
            )
            return result
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

def encode_image_for_chatgpt(image_path):
    """Encode image to base64 for ChatGPT API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class ChatGPTProcessor(ModelProcessor):
    def __init__(self, api_key, model_name):
        super().__init__(api_key=api_key, model_name=model_name)
        if openai is None:
            raise ImportError("openai package is required for ChatGPT evaluation. Install it with: pip install openai")
        openai.api_key = api_key
        self.model_name = model_name
    
    def process_image(self, image_path, prompt):
        try:
            image_base64 = encode_image_for_chatgpt(image_path)
            
            # Special prompt for gpt-5 model only (not other GPT models)
            if self.model_name == 'gpt-5':
                # Override prompt with special system prompt for gpt-5
                # This applies to both regular and user study images
                prompt = "You are a vision analysis assistant. You may only compare observable visual features and overall facial similarity. Do not infer identity. Taking into account the face aging, lighting different hair styles, wearing and not wearing of eye glasses or other accessory, do the task in the image. Only answer yes or no"
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ],
                }]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

def evaluate_images_in_directory_unified(
    directory_path, 
    processor, 
    model_name, 
    max_images=50,
    output_prefix="results"
):
    """
    Unified function to evaluate all images in directory using any model processor.
    
    Args:
        directory_path: Path to directory containing images
        processor: ModelProcessor instance (GeminiProcessor, QwenVLProcessor, or ChatGPTProcessor)
        model_name: Model name for output filename
        max_images: Maximum number of images to process
        output_prefix: Prefix for output CSV filename (e.g., "gemini", "qwenvl", "chatgpt")
    """
    total_images = 0
    yes_responses = 0
    
    # Create CSV file
    safe_model_name = model_name.replace('-', '_')
    csv_filename = os.path.join(
        os.path.dirname(directory_path), 
        f"{output_prefix}_results_{safe_model_name}.csv"
    )
    
    # Load already processed images
    processed_images = set()
    if os.path.exists(csv_filename):
        print(f"Loading existing results from {csv_filename}...")
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if 'filename' in row:
                    processed_images.add(row['filename'])
        print(f"Found {len(processed_images)} already processed images. Skipping them...")
    
    skipped_count = 0
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header only if file is new
        if len(processed_images) == 0:
            csv_writer.writerow(['filename', 'response', 'label'])
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if total_images >= max_images:
                    break
                    
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    filename = os.path.basename(image_path)
                    
                    # Skip if already processed
                    if filename in processed_images:
                        skipped_count += 1
                        if skipped_count % 100 == 0:
                            print(f"Skipped {skipped_count} already processed images...")
                        continue
                    
                    prompt = get_prompt_for_image(image_path, task_type="regular")
                    response = processor.process_image(image_path, prompt)
                    print(response)
                    
                    if response is not None:
                        total_images += 1
                        label = 'yes' if normalize_response(response) else 'no'
                        
                        csv_writer.writerow([filename, response, label])
                        csvfile.flush()  # Ensure data is written immediately
                        
                        if label == 'yes':
                            yes_responses += 1
                            print(f"YES: {image_path}")
                        else:
                            print(f"NO: {image_path}")
                        
                        accuracy = (yes_responses / total_images) * 100
                        print(f"Current accuracy: {accuracy:.2f}% ({yes_responses}/{total_images})")
            
            if total_images >= max_images:
                break
    
    # Final results
    print("\nFinal Results:")
    print(f"Total images processed: {total_images}")
    print(f"Total images skipped: {skipped_count}")
    print(f"Number of 'yes' responses: {yes_responses}")
    if total_images > 0:
        accuracy = (yes_responses / total_images) * 100
        print(f"Final accuracy: {accuracy:.2f}%")
    print(f"\nResults have been saved to: {csv_filename}")

def evaluate_user_study_images_unified(
    directory_path,
    processor,
    model_name,
    max_images=None,
    output_prefix="user_study"
):
    """
    Unified function to evaluate user study images using any model processor.
    
    Args:
        directory_path: Path to directory containing user study images
        processor: ModelProcessor instance
        model_name: Model name for output filename
        max_images: Maximum number of images to process (None for all)
        output_prefix: Prefix for output CSV filename (e.g., "user_study_gemini", "user_study_qwenvl", "user_study_chatgpt")
    
    Returns:
        dict: Statistics with yes_count, no_count, refuse_count, total, and rates
    """
    dataset_type = get_dataset_type_from_path(directory_path)
    
    total_images = 0
    yes_count = 0
    no_count = 0
    refuse_count = 0
    
    # Create CSV file
    safe_model_name = model_name.replace('-', '_')
    csv_filename = os.path.join(
        os.path.dirname(directory_path),
        f"{output_prefix}_results_{safe_model_name}.csv"
    )
    
    # Load already processed images
    processed_images = set()
    if os.path.exists(csv_filename):
        print(f"Loading existing results from {csv_filename}...")
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if 'filename' in row:
                    processed_images.add(row['filename'])
        print(f"Found {len(processed_images)} already processed images. Skipping them...")
    
    skipped_count = 0
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header only if file is new
        if len(processed_images) == 0:
            csv_writer.writerow(['filename', 'class', 'response', 'category'])
        
        for root, dirs, files in os.walk(directory_path):
            files = sorted([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            for file in files:
                if max_images is not None and total_images >= max_images:
                    break
                
                # Skip if already processed
                if file in processed_images:
                    skipped_count += 1
                    if skipped_count % 100 == 0:
                        print(f"Skipped {skipped_count} already processed images...")
                    continue
                    
                image_path = os.path.join(root, file)
                rel_path = os.path.relpath(image_path, directory_path)
                class_id = os.path.dirname(rel_path) if os.path.dirname(rel_path) else 'root'
                
                prompt = get_prompt_for_image(image_path, task_type="user_study")
                response = processor.process_image(image_path, prompt)
                print(response)
                
                if response is not None:
                    total_images += 1
                    category = categorize_response(response)
                    
                    if category == 'yes':
                        yes_count += 1
                    elif category == 'no':
                        no_count += 1
                    else:
                        refuse_count += 1
                    
                    csv_writer.writerow([file, class_id, response, category])
                    csvfile.flush()  # Ensure data is written immediately
                    
                    status_emoji = {'yes': '✓', 'no': '✗', 'refuse': '⚠'}[category]
                    print(f"{status_emoji} [{category.upper()}]: {image_path}")
                    
                    yes_rate = (yes_count / total_images) * 100 if total_images > 0 else 0
                    no_rate = (no_count / total_images) * 100 if total_images > 0 else 0
                    refuse_rate = (refuse_count / total_images) * 100 if total_images > 0 else 0
                    print(f"  Rates - Yes: {yes_rate:.2f}% | No: {no_rate:.2f}% | Refuse: {refuse_rate:.2f}% ({total_images} total)")
            
            if max_images is not None and total_images >= max_images:
                break
    
    # Calculate final rates
    yes_rate = (yes_count / total_images) * 100 if total_images > 0 else 0
    no_rate = (no_count / total_images) * 100 if total_images > 0 else 0
    refuse_rate = (refuse_count / total_images) * 100 if total_images > 0 else 0
    
    print("\n" + "="*60)
    print(f"FINAL RESULTS - User Study Evaluation ({output_prefix})")
    print("="*60)
    print(f"Total images processed: {total_images}")
    print(f"Total images skipped: {skipped_count}")
    print(f"\nResponse Counts:")
    print(f"  Yes:   {yes_count:4d} ({yes_rate:.2f}%)")
    print(f"  No:    {no_count:4d} ({no_rate:.2f}%)")
    print(f"  Refuse: {refuse_count:4d} ({refuse_rate:.2f}%)")
    print(f"\nResults saved to: {csv_filename}")
    print("="*60)
    
    return {
        'yes_count': yes_count,
        'no_count': no_count,
        'refuse_count': refuse_count,
        'total': total_images,
        'yes_rate': yes_rate,
        'no_rate': no_rate,
        'refuse_rate': refuse_rate
    }

# ===== Model-Specific Wrapper Functions (Using Unified Framework) =====

def evaluate_images_in_directory(directory_path, api_key, mllm_model_name, max_images=50):
    """Evaluate all images in the directory and its subdirectories (Gemini wrapper)"""
    processor = GeminiProcessor(api_key, mllm_model_name)
    return evaluate_images_in_directory_unified(
        directory_path, processor, mllm_model_name, max_images, "gemini"
    )


def evaluate_user_study_images(directory_path, api_key, max_images=None):
    """
    Evaluate user study images (positive/negative pairs) with Gemini and track Yes/No/Refuse rates.
    
    Args:
        directory_path: Path to directory containing user study images (organized by class folders)
        api_key: Gemini API key
        max_images: Maximum number of images to process (None for all)
    
    Returns:
        dict: Statistics with yes_count, no_count, refuse_count, total, and rates
    """
    processor = GeminiProcessor(api_key, 'gemini-2.0-flash')
    return evaluate_user_study_images_unified(
        directory_path, processor, 'gemini-2.0-flash', max_images, "user_study_gemini"
    )


def compute_metrics(ground_truth_path, prediction_path):
    # Read the CSV files
    ground_truth = pd.read_csv(ground_truth_path)
    predictions = pd.read_csv(prediction_path)
    
    # Ensure the column names are consistent
    ground_truth.columns = ['filename', 'label']
    predictions.columns = ['filename', 'prediction']

    ground_truth['extracted_filename'] = ground_truth['filename'].str.extract(r"(ID=\d+_T=\d+)")
    predictions['extracted_filename'] = predictions['filename'].str.extract(r"(ID=\d+_T=\d+)")
    # Merge the dataframes on filename
    merged = pd.merge(ground_truth, predictions, on='extracted_filename', how='inner')

    # print(len(merged))
    
    # Convert labels to binary (yes=1, no=0)
    merged['label'] = (merged['label'] == 'yes').astype(int)
    merged['prediction'] = (merged['prediction'] == 'yes').astype(int)
    
    # Calculate metrics
    TP = ((merged['label'] == 1) & (merged['prediction'] == 1)).sum()
    TN = ((merged['label'] == 0) & (merged['prediction'] == 0)).sum()
    FP = ((merged['label'] == 0) & (merged['prediction'] == 1)).sum()
    FN = ((merged['label'] == 1) & (merged['prediction'] == 0)).sum()
    
    # Calculate rates
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Sensitivity)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate (Specificity)
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # False Negative Rate
    
    # Print results
    print("\nConfusion Matrix Metrics:")
    print(f"TP: {TP}")
    print(f"TN: {TN}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    
 
    
    print("\nAttribute Accuracy:")
    AttAcc_F_Gemini= (merged['label'] == 1).sum() / len(merged)
    print(f"AttAcc_F_Gemini: {AttAcc_F_Gemini*100:.2f}")
    AttAcc_F_Curr = (merged['prediction'] == 1).sum() / len(merged)
    print(f"AttAcc_F_Curr: {AttAcc_F_Curr*100:.2f}")

    print(f"FPR: {FPR*100:.2f}")
    print(f"FNR: {FNR*100:.2f}")
    print(f"TPR: {TPR*100:.2f}")
    print(f"TNR: {TNR*100:.2f}")
    
    return merged

# QwenVL evaluation functions
def evaluate_images_in_directory_qwenvl(directory_path, server_url, mllm_model_name, max_images=50, api_name="/process_query"):
    """Evaluate all images in the directory and its subdirectories using QwenVL"""
    processor = QwenVLProcessor(server_url, mllm_model_name, api_name)
    return evaluate_images_in_directory_unified(
        directory_path, processor, mllm_model_name, max_images, "qwenvl"
    )

def evaluate_user_study_images_qwenvl(directory_path, server_url, mllm_model_name, max_images=None, api_name="/process_query"):
    """
    Evaluate user study images (positive/negative pairs) with QwenVL and track Yes/No/Refuse rates.
    
    Args:
        directory_path: Path to directory containing user study images (organized by class folders)
        server_url: QwenVL server URL
        mllm_model_name: MLLM model name (e.g., 'qwenvl-25-72b')
        max_images: Maximum number of images to process (None for all)
        api_name: API endpoint name (default: "/process_query")
    
    Returns:
        dict: Statistics with yes_count, no_count, refuse_count, total, and rates
    """
    processor = QwenVLProcessor(server_url, mllm_model_name, api_name)
    return evaluate_user_study_images_unified(
        directory_path, processor, mllm_model_name, max_images, "user_study_qwenvl"
    )

# ===== ChatGPT Functions =====

def evaluate_images_in_directory_chatgpt(directory_path, api_key, model_name, max_images=50):
    """Evaluate all images in the directory and its subdirectories using ChatGPT"""
    processor = ChatGPTProcessor(api_key, model_name)
    return evaluate_images_in_directory_unified(
        directory_path, processor, model_name, max_images, "chatgpt"
    )

def evaluate_user_study_images_chatgpt(directory_path, api_key, model_name, max_images=None):
    """
    Evaluate user study images (positive/negative pairs) with ChatGPT and track Yes/No/Refuse rates.
    
    Args:
        directory_path: Path to directory containing user study images (organized by class folders)
        api_key: OpenAI API key
        model_name: ChatGPT model name (e.g., 'gpt-4-vision-preview', 'gpt-5-chat-latest')
        max_images: Maximum number of images to process (None for all)
    
    Returns:
        dict: Statistics with yes_count, no_count, refuse_count, total, and rates
    """
    processor = ChatGPTProcessor(api_key, model_name)
    return evaluate_user_study_images_unified(
        directory_path, processor, model_name, max_images, "user_study_chatgpt"
    )