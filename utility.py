import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import google.generativeai as genai
from PIL import Image
import csv
import pandas as pd
# Your crop function (for CelebA)
def crop(x):
    return x[:, 55:55 + 108, 35:35 + 108]

# Create processor that applies crop to PIL image
processor = transforms.Compose([
    transforms.ToTensor(),        # PIL → Tensor
    transforms.Lambda(crop),      # Crop the tensor
    transforms.ToPILImage(),      # Tensor → PIL
    transforms.Resize((224, 224))   # Resize final output
])

def create_concatenated_images_with_labels(private_image_folder, mi_folder, output_folder):
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
        
        # Process each image in mi_folder
        for mi_image_path in mi_images:
            # Open the mi image and resize to 224x224
            mi_image = Image.open(mi_image_path).resize((224, 224))

            # Select 4 random images from the private images
            selected_private_images = random.sample(private_images, 4)
            # private_images_resized = [Image.open(img).resize((224, 224)) for img in selected_private_images]
            private_images_resized = []

            for img in selected_private_images:
                image = Image.open(img)
                if any(attack in mi_folder for attack in ["KEDMI", "LOMMA", "PLGMI"]):
                    image = processor(image)
                else:
                    image = image.resize((224, 224))
                private_images_resized.append(image)
                
            # Calculate the width and height for the new concatenated image
            total_width = (224 * 5) + white_space_width
            total_height = 224 + text_space_height + question_space_height
            concatenated_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))  # White background
            
            # Paste the mi image and private images with white space in between
            concatenated_image.paste(mi_image, (0, 0))
            x_offset = 224 + white_space_width
            for private_image in private_images_resized:
                concatenated_image.paste(private_image, (x_offset, 0))
                x_offset += 224
            
            # Draw text labels and question
            draw = ImageDraw.Draw(concatenated_image)
            # Draw "Image A" label
            draw.text((112, 224 + 5 + 20), "Image A", fill="black", font=font, anchor="mm")  # Centered below first image
            
            # Draw "Set of Images B" label
            draw.text((224 * 3 + white_space_width / 2, 224 + 5 + 20), "Set of Images B", fill="black", font=font, anchor="mm")  # Centered below private images
            
            # Draw question in red
            draw.text((total_width / 2, 224 + text_space_height + 5 + 20), 
                      "Does Image A depict the same individual as the images in Set B?", 
                      fill="red", font=question_font, anchor="mm")

            # Define the output image path with the same name as the mi image
            output_image_path = os.path.join(output_subfolder, os.path.basename(mi_image_path))
            concatenated_image.save(output_image_path)


def normalize_response(response):
    """Normalize the response to handle different cases of 'yes'"""
    response = response.strip().lower()
    return response == 'yes'

def process_image_with_gemini(image_path, model):
    """Process a single image with Gemini and return the response"""
    try:
        image = Image.open(image_path)
        response = model.generate_content([
            "You are an expert in face recognition. Taking into account the face aging, lighting, different hair styles, wearing and not wearing of eye glasses or other accessory, do the task in the image. Only answer yes or no",
            image
        ])
        return response.text
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def evaluate_images_in_directory(directory_path, api_key, max_images=50):
    """Evaluate all images in the directory and its subdirectories"""
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    total_images = 0
    yes_responses = 0
    
    # Create CSV file 
    csv_filename = os.path.join(os.path.dirname(directory_path), f"gemini_results.csv")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['filename', 'label'])
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if total_images >= max_images:
                    break
                    
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    response = process_image_with_gemini(image_path, model)
                    
                    if response is not None:
                        total_images += 1
                        label = 'yes' if normalize_response(response) else 'no'
                        
                        # Write to CSV
                        csv_writer.writerow([os.path.basename(image_path), label])
                        
                        if label == 'yes':
                            yes_responses += 1
                            print(f"YES: {image_path}")
                        else:
                            print(f"NO: {image_path}")
                            
                        
                        # Calculate and print real-time accuracy
                        accuracy = (yes_responses / total_images) * 100
                        print(f"Current accuracy: {accuracy:.2f}% ({yes_responses}/{total_images})")
            
            if total_images >= max_images:
                break
    
    # Final results
    print("\nFinal Results:")
    print(f"Total images processed: {total_images}")
    print(f"Number of 'yes' responses: {yes_responses}")
    print(f"Final accuracy: {accuracy:.2f}%")
    print(f"\nResults have been saved to: {csv_filename}")


def list_image_files(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def create_prediction_csv(base_path):
    # Create output CSV file
    output_file = os.path.join(base_path, 'prediction.csv')
    
    # Get all files from Positive and Negative folders
    positive_path = os.path.join(base_path, 'Positive')
    negative_path = os.path.join(base_path, 'Negative')
    
    # Prepare data for CSV
    data = []
    
    # Process Positive folder
    if os.path.exists(positive_path):
        image_files = list_image_files(positive_path)
        for filename in image_files:
            if filename.endswith('.png'):  # Only process PNG files
                data.append([os.path.basename(filename), 'yes'])
    
    # Process Negative folder
    if os.path.exists(negative_path):
        image_files = list_image_files(negative_path)
        for filename in image_files:
            if filename.endswith('.png'):  # Only process PNG files
                data.append([os.path.basename(filename), 'no'])
    # Sort data by filename to maintain consistent order
    data.sort(key=lambda x: x[0])
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'prediction'])  # Write header
        writer.writerows(data)  # Write data rows
    
    print(f"Created prediction CSV at: {output_file}")
    print(f"Total entries: {len(data)}")

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