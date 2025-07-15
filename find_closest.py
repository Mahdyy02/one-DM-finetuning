import os
import shutil
import torch
from PIL import Image
import open_clip
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from difflib import SequenceMatcher
import re

# --- CONFIG ---
DST_FOLDER = "girl"
QUERY_IMAGE = "283_girl.png"  # <-- Replace with your query image path
CLIP_WEIGHT = 0.5
OCR_WEIGHT = 0.5

# --- LOAD MODELS ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# CLIP
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
model = model.to(device).eval()
# TrOCR
ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = ocr_processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = ocr_model.generate(pixel_values)
        generated_text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        cleaned_text = re.sub(r'\s+', ' ', generated_text.strip().lower())
        return cleaned_text
    except Exception as e:
        print(f"OCR failed for {image_path}: {e}")
        return ""

def calculate_text_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()

@torch.no_grad()
def get_clip_embedding(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    return model.encode_image(image).squeeze().cpu()

def find_closest_image_with_ocr(query_path, folder, clip_weight=0.5, ocr_weight=0.5):
    # Get query features
    query_emb = get_clip_embedding(query_path)
    query_text = extract_text_from_image(query_path)
    print(f"Query OCR: '{query_text}'")
    best_score = -float('inf')
    best_path = None
    best_clip = 0.0
    best_ocr = 0.0
    for fname in os.listdir(folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, fname)
            db_emb = get_clip_embedding(img_path)
            db_text = extract_text_from_image(img_path)
            clip_sim = torch.nn.functional.cosine_similarity(query_emb, db_emb, dim=0).item()
            ocr_sim = calculate_text_similarity(query_text, db_text)
            combined = (clip_weight * clip_sim) + (ocr_weight * ocr_sim)
            print(f"{fname}: Combined={combined:.4f} (CLIP={clip_sim:.4f}, OCR={ocr_sim:.4f})")
            if combined > best_score:
                best_score = combined
                best_path = img_path
                best_clip = clip_sim
                best_ocr = ocr_sim
    return best_path, best_score, best_clip, best_ocr

if __name__ == "__main__":
    # Replace QUERY_IMAGE with your actual query image path
    if not os.path.isfile(QUERY_IMAGE):
        print(f"Query image not found: {QUERY_IMAGE}")
    else:
        best_path, best_score, best_clip, best_ocr = find_closest_image_with_ocr(
            QUERY_IMAGE, DST_FOLDER, CLIP_WEIGHT, OCR_WEIGHT)
        print(f"\nClosest image: {best_path}")
        print(f"Combined score: {best_score:.4f}")
        print(f"CLIP score: {best_clip:.4f}")
        print(f"OCR score: {best_ocr:.4f}") 