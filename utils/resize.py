import os
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def resize_image_proportional_height(image, target_height=64):
    """
    Resize image to target height while maintaining aspect ratio
    Width will be proportional to original aspect ratio
    
    Args:
        image: Input image (numpy array)
        target_height: Target height in pixels (default: 64)
    
    Returns:
        Resized image as numpy array
    """
    original_height, original_width = image.shape[:2]
    
    # Calculate aspect ratio
    aspect_ratio = original_width / original_height
    
    # Calculate new width maintaining aspect ratio
    new_width = int(target_height * aspect_ratio)
    
    # Resize image
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    
    return resized

def get_image_extensions():
    """Return common image file extensions"""
    return {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

def process_image(input_path, output_path, target_height=64, quality=95):
    """
    Process a single image - resize to target height with proportional width
    
    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        target_height: Target height in pixels (default: 64)
        quality: JPEG quality (default: 95)
    """
    try:
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Warning: Could not read image {input_path}")
            return False, None
        
        original_height, original_width = image.shape[:2]
        
        # Resize image maintaining aspect ratio
        resized_image = resize_image_proportional_height(image, target_height)
        
        # Get final dimensions
        final_height, final_width = resized_image.shape[:2]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the resized image
        file_ext = os.path.splitext(output_path)[1].lower()
        if file_ext in ['.jpg', '.jpeg']:
            cv2.imwrite(output_path, resized_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif file_ext == '.png':
            cv2.imwrite(output_path, resized_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(output_path, resized_image)
        
        # Verify final size
        saved_image = cv2.imread(output_path)
        if saved_image is not None:
            saved_height, saved_width = saved_image.shape[:2]
            if saved_height != target_height:
                print(f"Warning: Height mismatch for {input_path}: expected {target_height}, got {saved_height}")
        
        return True, (original_width, original_height, final_width, final_height)
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False, None

def get_image_info(folder_path):
    """Get information about images in folder"""
    image_extensions = get_image_extensions()
    total_images = 0
    size_info = {}
    aspect_ratios = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file.lower())[1] in image_extensions:
                total_images += 1
                file_path = os.path.join(root, file)
                try:
                    img = cv2.imread(file_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        size_key = f"{w}x{h}"
                        size_info[size_key] = size_info.get(size_key, 0) + 1
                        aspect_ratios.append(w/h)
                except:
                    pass
    
    return total_images, size_info, aspect_ratios

def process_folder_structure(input_folder, output_folder, target_height=64, quality=95):
    """
    Process entire folder structure - resize all images to target height with proportional width
    
    Args:
        input_folder: Root folder containing subfolders with images
        output_folder: Root folder for resized images
        target_height: Target height in pixels (default: 64)
        quality: JPEG quality (default: 95)
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get image information
    total_images, size_info, aspect_ratios = get_image_info(input_folder)
    
    print(f"Found {total_images} images to resize")
    print(f"Target height: {target_height} pixels (width will be proportional)")
    print(f"Original image sizes found:")
    for size, count in sorted(size_info.items()):
        print(f"  {size}: {count} images")
    
    if aspect_ratios:
        min_aspect = min(aspect_ratios)
        max_aspect = max(aspect_ratios)
        avg_aspect = sum(aspect_ratios) / len(aspect_ratios)
        
        print(f"\nAspect ratio analysis:")
        print(f"  Min aspect ratio: {min_aspect:.2f} (narrowest)")
        print(f"  Max aspect ratio: {max_aspect:.2f} (widest)")
        print(f"  Average aspect ratio: {avg_aspect:.2f}")
        print(f"  Expected width range: {int(target_height * min_aspect)}-{int(target_height * max_aspect)} pixels")
    
    # Get image extensions
    image_extensions = get_image_extensions()
    
    # Process all images
    processed_count = 0
    failed_count = 0
    size_changes = []
    
    with tqdm(total=total_images, desc="Resizing images") as pbar:
        for root, dirs, files in os.walk(input_folder):
            # Calculate relative path from input folder
            rel_path = os.path.relpath(root, input_folder)
            
            # Create corresponding output directory
            if rel_path == '.':
                output_root = output_folder
            else:
                output_root = os.path.join(output_folder, rel_path)
            
            # Process each image file
            for file in files:
                file_ext = os.path.splitext(file.lower())[1]
                if file_ext in image_extensions:
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(output_root, file)
                    
                    # Process the image
                    success, size_info = process_image(input_path, output_path, target_height, quality)
                    if success:
                        processed_count += 1
                        if size_info:
                            size_changes.append(size_info)
                    else:
                        failed_count += 1
                    
                    pbar.update(1)
    
    print(f"\nResizing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Failed to process: {failed_count} images")
    print(f"All images resized to height: {target_height} pixels")
    
    if size_changes:
        print(f"\nSize change summary:")
        original_widths = [s[0] for s in size_changes]
        final_widths = [s[2] for s in size_changes]
        
        print(f"  Original width range: {min(original_widths)}-{max(original_widths)} pixels")
        print(f"  Final width range: {min(final_widths)}-{max(final_widths)} pixels")
        print(f"  Average width reduction: {(sum(original_widths)/len(original_widths)):.0f} → {(sum(final_widths)/len(final_widths)):.0f} pixels")
    
    print(f"Output saved to: {output_folder}")

def preview_resize(input_folder, target_height=64):
    """Show preview of proportional height resize on first few images"""
    image_extensions = get_image_extensions()
    previews = []
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if os.path.splitext(file.lower())[1] in image_extensions:
                input_path = os.path.join(root, file)
                
                # Load original image
                original = cv2.imread(input_path)
                if original is not None:
                    orig_h, orig_w = original.shape[:2]
                    
                    # Resize with proportional width
                    resized = resize_image_proportional_height(original, target_height)
                    new_h, new_w = resized.shape[:2]
                    
                    print(f"Preview: {os.path.basename(input_path)}")
                    print(f"  Original: {orig_w}x{orig_h}")
                    print(f"  Resized:  {new_w}x{new_h}")
                    print(f"  Aspect ratio preserved: {orig_w/orig_h:.3f} → {new_w/new_h:.3f}")
                    
                    # Create side-by-side comparison
                    # Scale original for display if too large
                    if orig_h > 200:
                        display_scale = 200 / orig_h
                        display_orig = cv2.resize(original, (int(orig_w * display_scale), 200))
                    else:
                        display_orig = original.copy()
                    
                    # Scale resized for display
                    display_scale = 200 / new_h
                    display_resized = cv2.resize(resized, (int(new_w * display_scale), 200))
                    
                    # Create comparison image
                    max_width = max(display_orig.shape[1], display_resized.shape[1])
                    
                    # Pad images to same width for display
                    if display_orig.shape[1] < max_width:
                        pad_width = max_width - display_orig.shape[1]
                        display_orig = cv2.copyMakeBorder(display_orig, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=[255,255,255])
                    
                    if display_resized.shape[1] < max_width:
                        pad_width = max_width - display_resized.shape[1]
                        display_resized = cv2.copyMakeBorder(display_resized, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=[255,255,255])
                    
                    # Stack vertically
                    combined = np.vstack([display_orig, display_resized])
                    
                    # Add labels
                    label_img = np.ones((60, combined.shape[1], 3), dtype=np.uint8) * 255
                    cv2.putText(label_img, f'Original: {orig_w}x{orig_h}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
                    cv2.putText(label_img, f'Resized: {new_w}x{new_h} (height={target_height})', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
                    
                    if len(combined.shape) == 2:
                        combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
                    
                    final_preview = np.vstack([label_img, combined])
                    
                    cv2.imshow(f'Proportional Height Resize Preview', final_preview)
                    key = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                    if key == ord('q'):
                        break
                    elif key == ord('n'):
                        continue
                    else:
                        break
                        
                if len(previews) >= 3:  # Show max 3 previews
                    break
            else:
                continue
        else:
            continue
        break

def main():
    parser = argparse.ArgumentParser(description='Resize all images in folder structure to fixed height with proportional width')
    parser.add_argument('--input', '-i', required=True, help='Input folder path')
    parser.add_argument('--output', '-o', required=True, help='Output folder path')
    parser.add_argument('--height', '-hh', type=int, default=64, help='Target height in pixels (default: 64)')
    parser.add_argument('--quality', '-q', type=int, default=95, help='JPEG quality (default: 95)')
    parser.add_argument('--preview', action='store_true', 
                       help='Show preview of proportional resize before processing')
    parser.add_argument('--info', action='store_true',
                       help='Show information about images in input folder')
    
    args = parser.parse_args()
    
    # Info mode
    if args.info:
        total_images, size_info, aspect_ratios = get_image_info(args.input)
        print(f"Image information for: {args.input}")
        print(f"Total images: {total_images}")
        print("Size distribution:")
        for size, count in sorted(size_info.items()):
            print(f"  {size}: {count} images")
        
        if aspect_ratios:
            min_aspect = min(aspect_ratios)
            max_aspect = max(aspect_ratios)
            avg_aspect = sum(aspect_ratios) / len(aspect_ratios)
            
            print(f"\nAspect ratio analysis:")
            print(f"  Min aspect ratio: {min_aspect:.2f}")
            print(f"  Max aspect ratio: {max_aspect:.2f}")
            print(f"  Average aspect ratio: {avg_aspect:.2f}")
            print(f"  With height {args.height}, widths will range: {int(args.height * min_aspect)}-{int(args.height * max_aspect)} pixels")
        return
    
    # Preview mode
    if args.preview:
        print("Preview mode: Showing proportional height resize...")
        preview_resize(args.input, args.height)
        
        choice = input("Continue with processing all images? (y/n): ")
        if choice.lower() != 'y':
            return
    
    # Process all images
    process_folder_structure(
        args.input, 
        args.output, 
        args.height, 
        args.quality
    )

if __name__ == "__main__":
    main()