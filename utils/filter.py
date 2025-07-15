import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def apply_high_pass_filter(image, kernel_size=3, method='gaussian', binary_output=True, threshold_value=127, noise_reduction=True):
    """
    Apply filtering to preserve text and remove background noise
    
    Args:
        image: Input image (numpy array)
        kernel_size: Size of the kernel for filtering
        method: 'gaussian', 'laplacian', or 'sobel'
        binary_output: Whether to convert result to black and white (binary)
        threshold_value: Threshold value for binary conversion (0-255)
        noise_reduction: Whether to apply noise reduction techniques
    
    Returns:
        Filtered image with text preserved and background noise removed
    """
    if len(image.shape) == 3:
        # Convert to grayscale if colored
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Pre-processing for noise reduction
    if noise_reduction:
        # 1. Bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Morphological opening to remove small noise
        kernel_noise = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_noise)
    
    if method == 'gaussian':
        # For text preservation, use unsharp masking instead of high-pass
        # Apply Gaussian blur and then enhance contrast
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        # Unsharp masking: original + (original - blurred) * amount
        processed = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        processed = np.clip(processed, 0, 255).astype(np.uint8)
    
    elif method == 'laplacian':
        # Use Laplacian for edge detection but preserve text
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(laplacian)
        # Combine original with edge information
        processed = cv2.addWeighted(gray, 0.8, laplacian, 0.2, 0)
    
    elif method == 'sobel':
        # Use Sobel for edge detection but preserve text
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.uint8(sobel)
        # Combine original with edge information
        processed = cv2.addWeighted(gray, 0.8, sobel, 0.2, 0)
    
    else:
        raise ValueError("Method must be 'gaussian', 'laplacian', or 'sobel'")
    
    # Post-processing noise reduction
    if noise_reduction:
        # Additional noise cleanup while preserving text
        kernel_clean = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_clean)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_clean)
    
    # Convert to black and white if requested
    if binary_output:
        if noise_reduction:
            # Use adaptive thresholding for better text preservation
            processed = cv2.adaptiveThreshold(
                processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            # Apply simple threshold to create pure black and white image
            _, processed = cv2.threshold(processed, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Invert the image so text is black and background is white
        processed = cv2.bitwise_not(processed)
        
        # Final cleanup to remove small noise while preserving text
        if noise_reduction:
            # Remove small white noise spots (opening)
            kernel_final = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_final)
            
            # Fill small holes in text (closing)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_final)
    
    return processed

def get_image_extensions():
    """Return common image file extensions"""
    return {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

def process_image(input_path, output_path, kernel_size=3, method='gaussian', preserve_format=True, binary_output=True, threshold_value=127, noise_reduction=True):
    """
    Process a single image with high pass filter
    
    Args:
        input_path: Path to input image
        output_path: Path to save filtered image
        kernel_size: Size of the kernel for filtering
        method: Filtering method
        preserve_format: Whether to keep original format or save as PNG
        binary_output: Whether to convert result to black and white
        threshold_value: Threshold value for binary conversion
        noise_reduction: Whether to apply noise reduction techniques
    """
    try:
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Warning: Could not read image {input_path}")
            return False
        
        # Get original dimensions
        original_height, original_width = image.shape[:2]
        
        # Apply high pass filter
        filtered_image = apply_high_pass_filter(image, kernel_size, method, binary_output, threshold_value, noise_reduction)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the filtered image
        if preserve_format:
            # Keep original format
            cv2.imwrite(output_path, filtered_image)
        else:
            # Save as PNG for consistency
            base_name = os.path.splitext(output_path)[0]
            png_path = base_name + '.png'
            cv2.imwrite(png_path, filtered_image)
        
        # Verify dimensions are preserved
        saved_image = cv2.imread(output_path if preserve_format else png_path)
        if saved_image is not None:
            saved_height, saved_width = saved_image.shape[:2]
            if saved_height != original_height or saved_width != original_width:
                print(f"Warning: Size mismatch for {input_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_folder_structure(input_folder, output_folder, kernel_size=3, method='gaussian', preserve_format=True, binary_output=True, threshold_value=127, noise_reduction=True):
    """
    Process entire folder structure with high pass filter
    
    Args:
        input_folder: Root folder containing subfolders with images
        output_folder: Root folder for filtered images
        kernel_size: Size of the kernel for filtering
        method: Filtering method ('gaussian', 'laplacian', 'sobel')
        preserve_format: Whether to keep original image format
        binary_output: Whether to convert result to black and white
        threshold_value: Threshold value for binary conversion
        noise_reduction: Whether to apply noise reduction techniques
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get image extensions
    image_extensions = get_image_extensions()
    
    # Count total images for progress bar
    total_images = 0
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if os.path.splitext(file.lower())[1] in image_extensions:
                total_images += 1
    
    print(f"Found {total_images} images to process")
    print(f"Using {method} high pass filter with kernel size {kernel_size}")
    if binary_output:
        threshold_info = "adaptive thresholding" if noise_reduction else f"threshold {threshold_value}"
        print(f"Output will be binary (black and white) with {threshold_info}")
    if noise_reduction:
        print("Noise reduction is enabled")
    
    # Process all images
    processed_count = 0
    failed_count = 0
    
    with tqdm(total=total_images, desc="Processing images") as pbar:
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
                    if process_image(input_path, output_path, kernel_size, method, preserve_format, binary_output, threshold_value, noise_reduction):
                        processed_count += 1
                    else:
                        failed_count += 1
                    
                    pbar.update(1)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Failed to process: {failed_count} images")
    print(f"Output saved to: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='Apply high pass filter to images in folder structure')
    parser.add_argument('--input', '-i', required=True, help='Input folder path')
    parser.add_argument('--output', '-o', required=True, help='Output folder path')
    parser.add_argument('--kernel_size', '-k', type=int, default=3, help='Kernel size for filtering (default: 3)')
    parser.add_argument('--method', '-m', choices=['gaussian', 'laplacian', 'sobel'], 
                       default='gaussian', help='High pass filter method (default: gaussian)')
    parser.add_argument('--preserve_format', action='store_true', 
                       help='Keep original image format (default: False, saves as PNG)')
    parser.add_argument('--preview', action='store_true', 
                       help='Show preview of first image before processing all')
    parser.add_argument('--grayscale', action='store_true',
                       help='Output grayscale instead of black and white (binary)')
    parser.add_argument('--threshold', '-t', type=int, default=127,
                       help='Threshold value for black/white conversion (0-255, default: 127)')
    parser.add_argument('--no_noise_reduction', action='store_true',
                       help='Disable noise reduction techniques (faster but noisier)')
    
    args = parser.parse_args()
    
    # Determine if output should be binary (black and white) or grayscale
    binary_output = not args.grayscale
    noise_reduction = not args.no_noise_reduction
    
    # Preview mode
    if args.preview:
        print("Preview mode: Processing first image found...")
        image_extensions = get_image_extensions()
        
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if os.path.splitext(file.lower())[1] in image_extensions:
                    input_path = os.path.join(root, file)
                    
                    # Load and show original
                    original = cv2.imread(input_path)
                    if original is not None:
                        # Apply filter
                        filtered = apply_high_pass_filter(original, args.kernel_size, args.method, binary_output, args.threshold, noise_reduction)
                        
                        # Display side by side
                        if binary_output:
                            # For binary images, convert to 3-channel for display
                            filtered_display = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
                        else:
                            filtered_display = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
                        
                        combined = np.hstack((original, filtered_display))
                        noise_status = "with noise reduction" if noise_reduction else "without noise reduction"
                        window_title = f'Original (Left) vs Filtered {"Binary" if binary_output else "Grayscale"} {noise_status} (Right)'
                        cv2.imshow(window_title, combined)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        
                        choice = input("Continue with processing all images? (y/n): ")
                        if choice.lower() != 'y':
                            return
                        break
            else:
                continue
            break
    
    # Process all images
    process_folder_structure(
        args.input, 
        args.output, 
        args.kernel_size, 
        args.method, 
        args.preserve_format,
        binary_output,
        args.threshold,
        noise_reduction
    )

if __name__ == "__main__":
    main()