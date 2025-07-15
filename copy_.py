import os
import shutil

SRC_FOLDER = "Generated/English/oov_u"
DST_FOLDER = "girl"

# Create the destination folder if it doesn't exist
os.makedirs(DST_FOLDER, exist_ok=True)

count = 0
for root, dirs, files in os.walk(SRC_FOLDER):
    for fname in files:
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            src_path = os.path.join(root, fname)
            # Optionally, prefix the filename with the subfolder name to avoid overwriting
            subfolder = os.path.relpath(root, SRC_FOLDER)
            dst_fname = f"{subfolder.replace(os.sep, '_')}_{fname}" if subfolder != '.' else fname
            dst_path = os.path.join(DST_FOLDER, dst_fname)
            shutil.copy(src_path, dst_path)
            count += 1

print(f"Copied {count} images from all subfolders of {SRC_FOLDER} to {DST_FOLDER}")