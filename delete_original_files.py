import os
import argparse
from pathlib import Path

def delete_original_files(folder_path, preview=False):
    """
    Delete all files ending with '_original' in a folder and its subfolders
    
    Args:
        folder_path: Path to the folder to search
        preview: If True, only show what would be deleted without actually deleting
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    # Find all files ending with '_original'
    original_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if '_original' in file:
                file_path = os.path.join(root, file)
                original_files.append(file_path)
    
    if not original_files:
        print("No files ending with '_original' found.")
        return
    
    print(f"Found {len(original_files)} files ending with '_original':")
    
    # Show the files that would be deleted
    for i, file_path in enumerate(original_files[:10]):  # Show first 10
        rel_path = os.path.relpath(file_path, folder_path)
        print(f"  {rel_path}")
    
    if len(original_files) > 10:
        print(f"  ... and {len(original_files) - 10} more files")
    
    if preview:
        print("\nPreview mode - no files were deleted.")
        return
    
    # Ask for confirmation
    confirm = input(f"\nDelete all {len(original_files)} files? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Delete the files
    deleted_count = 0
    failed_count = 0
    
    for file_path in original_files:
        try:
            os.remove(file_path)
            deleted_count += 1
            print(f"Deleted: {os.path.relpath(file_path, folder_path)}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
            failed_count += 1
    
    print(f"\nDeletion complete!")
    print(f"Successfully deleted: {deleted_count} files")
    if failed_count > 0:
        print(f"Failed to delete: {failed_count} files")

def main():
    parser = argparse.ArgumentParser(description='Delete all files ending with "_original" in a folder')
    parser.add_argument('--input', '-i', required=True, help='Input folder path')
    parser.add_argument('--preview', action='store_true', help='Preview files to be deleted without actually deleting them')
    
    args = parser.parse_args()
    
    print(f"Delete Original Files")
    print(f"Input folder: {args.input}")
    print(f"Mode: {'Preview' if args.preview else 'Delete'}")
    print("-" * 50)
    
    delete_original_files(args.input, preview=args.preview)

if __name__ == "__main__":
    main()
