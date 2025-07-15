import os
import torch
import open_clip
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from difflib import SequenceMatcher
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load TrOCR model
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
model = model.to(device).eval()

def extract_text_from_image(image_path, debug=False):
    """Extract text from a handwritten image using TrOCR."""
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values.to(device)

        generated_ids = trocr_model.generate(pixel_values)
        generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        cleaned_text = re.sub(r'\s+', ' ', generated_text.strip().lower())

        if debug:
            print(f"OCR for {os.path.basename(image_path)}: '{cleaned_text}'")

        return cleaned_text
    except Exception as e:
        if debug:
            print(f"OCR failed for {image_path}: {e}")
        return ""


def calculate_text_similarity(text1, text2):
    """Calculate similarity between two text strings."""
    if not text1 or not text2:
        return 0.0
    
    # Use SequenceMatcher to get similarity ratio
    similarity = SequenceMatcher(None, text1, text2).ratio()
    return similarity

@torch.no_grad()
def get_embedding(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    return model.encode_image(image).squeeze().cpu()

def load_or_create_embeddings(folder, cache_path="clip_image_db.pt", debug_ocr=False):
    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=False)
        return data['embeddings'], data['paths'], data.get('texts', [])
    
    embeddings = []
    paths = []
    texts = []
    
    print("Processing images and extracting text...")
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    for i, fname in enumerate(files):
        img_path = os.path.join(folder, fname)
        try:
            # Get CLIP embedding
            emb = get_embedding(img_path)
            embeddings.append(emb)
            paths.append(img_path)
            
            # Extract text using OCR
            text = extract_text_from_image(img_path, debug=debug_ocr)
            texts.append(text)
            
            print(f"Processed {i+1}/{len(files)}: {fname}")
            if text:
                print(f"  Extracted text: {text[:100]}...")
            else:
                print(f"  No text extracted")
                
        except Exception as e:
            print(f"Skipped {img_path}: {e}")
    
    if embeddings:
        embeddings = torch.stack(embeddings)
        torch.save({
            'embeddings': embeddings, 
            'paths': paths, 
            'texts': texts
        }, cache_path)
    
    # Print OCR statistics
    non_empty_texts = [t for t in texts if t.strip()]
    print(f"\nOCR Statistics:")
    print(f"Total images processed: {len(texts)}")
    print(f"Images with text extracted: {len(non_empty_texts)}")
    print(f"OCR success rate: {len(non_empty_texts)/len(texts)*100:.1f}%")
    
    return embeddings, paths, texts

def find_closest_image_with_ocr(query_path, db_embeddings, image_paths, db_texts, clip_weight=0.5, ocr_weight=0.5):
    """
    Find closest image using combined CLIP and OCR similarity.
    
    Args:
        query_path: Path to query image
        db_embeddings: Database of CLIP embeddings
        image_paths: List of image paths
        db_texts: List of extracted texts from database images
        clip_weight: Weight for CLIP similarity (default: 0.8)
        ocr_weight: Weight for OCR similarity (default: 0.2)
    """
    # Get CLIP embedding for query
    query_embedding = get_embedding(query_path)
    clip_similarities = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), db_embeddings)
    
    # Get OCR text for query
    query_text = extract_text_from_image(query_path, debug=True)
    print(f"Query image text: '{query_text}'" if query_text else "Query image text: [No text found]")
    
    # Calculate combined similarities
    combined_similarities = []
    for i, (clip_sim, db_text) in enumerate(zip(clip_similarities, db_texts)):
        # Calculate OCR similarity
        ocr_sim = calculate_text_similarity(query_text, db_text)
        
        # Combine similarities with weights
        combined_sim = (clip_weight * clip_sim.item()) + (ocr_weight * ocr_sim)
        combined_similarities.append(combined_sim)
    
    # Find best match
    best_idx = max(range(len(combined_similarities)), key=lambda i: combined_similarities[i])
    
    return (image_paths[best_idx], 
            combined_similarities[best_idx], 
            clip_similarities[best_idx].item(), 
            calculate_text_similarity(query_text, db_texts[best_idx]))

def search_similar_image(query_image_path, folder_path, cache_path=None, clip_weight=0.5, ocr_weight=0.5, debug_ocr=False):
    """
    Find the most similar image in a folder to the query image using CLIP + OCR.
    
    Args:
        query_image_path: Path to the image you want to find similar images for
        folder_path: Path to the folder containing images to search through
        cache_path: Optional path for caching embeddings (default: uses folder name)
        clip_weight: Weight for CLIP similarity (default: 0.8)
        ocr_weight: Weight for OCR similarity (default: 0.2)
        debug_ocr: Show detailed OCR debugging info
    
    Returns:
        tuple: (path_to_most_similar_image, combined_score, clip_score, ocr_score)
    """
    # Create cache path based on folder name if not provided
    if cache_path is None:
        folder_name = os.path.basename(folder_path.rstrip('/\\'))
        cache_path = f"clip_ocr_cache_{folder_name}.pt"
    
    # Load or create embeddings for the folder
    print(f"Loading embeddings and OCR data for folder: {folder_path}")
    db_embeddings, image_paths, db_texts = load_or_create_embeddings(folder_path, cache_path, debug_ocr)
    
    if len(image_paths) == 0:
        print("No images found in the folder!")
        return None, 0.0, 0.0, 0.0
    
    print(f"Searching through {len(image_paths)} images...")
    
    # Find the most similar image
    best_match, combined_score, clip_score, ocr_score = find_closest_image_with_ocr(
        query_image_path, db_embeddings, image_paths, db_texts, clip_weight, ocr_weight
    )
    
    print(f"\nBest match: {best_match}")
    print(f"Combined score: {combined_score:.4f}")
    print(f"CLIP score: {clip_score:.4f}")
    print(f"OCR score: {ocr_score:.4f}")
    
    return best_match, combined_score, clip_score, ocr_score

def get_top_matches(query_image_path, folder_path, top_k=5, cache_path=None, clip_weight=1, ocr_weight=0):
    """
    Get top K most similar images.
    
    Returns:
        list: List of tuples (path, combined_score, clip_score, ocr_score)
    """
    if cache_path is None:
        folder_name = os.path.basename(folder_path.rstrip('/\\'))
        cache_path = f"clip_ocr_cache_{folder_name}.pt"
    
    db_embeddings, image_paths, db_texts = load_or_create_embeddings(folder_path, cache_path)
    
    if len(image_paths) == 0:
        return []
    
    # Get query data
    query_embedding = get_embedding(query_image_path)
    query_text = extract_text_from_image(query_image_path)
    
    # Calculate all similarities
    clip_similarities = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), db_embeddings)
    
    results = []
    for i, (clip_sim, db_text, img_path) in enumerate(zip(clip_similarities, db_texts, image_paths)):
        ocr_sim = calculate_text_similarity(query_text, db_text)
        combined_sim = (clip_weight * clip_sim.item()) + (ocr_weight * ocr_sim)

        results.append((img_path, combined_sim, clip_sim.item(), ocr_sim))
    
    # Sort by combined similarity and return top K
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# Example usage:
if __name__ == "__main__":
    # Example usage - replace with your actual paths
    query_image = fr"Generated\English\oov_u\169\instead.png"
    search_folder = fr"data\IAM64-new\test\169"

    # played_0 = extract_text_from_image(fr"Generated\English\oov_u\169\Instead.png", debug=False)
    # played_1 = extract_text_from_image(fr"data\IAM64-new\test\169\c04-139-04-05.png", debug=False)
    # print(f"Similarity between generated {played_0} and real {played_1} is {calculate_text_similarity(played_0, played_1)}") 

    
    # Top 5 matches
    print("\n" + "="*50)
    print("TOP 5 MATCHES:")
    top_matches = get_top_matches(query_image, search_folder, top_k=5, clip_weight=0.5, ocr_weight=0.5)
    for i, (path, combined, clip, ocr) in enumerate(top_matches, 1):
        print(f"{i}. {os.path.basename(path)}")
        print(f"   Combined: {combined:.4f} (CLIP: {clip:.4f}, OCR: {ocr:.4f})")