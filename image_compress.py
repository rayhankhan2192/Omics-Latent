from PIL import Image
import os
import math

def fit_tiff_to_size(folder_path, target_mb=9.5):
    # We aim slightly under 10MB to be safe (9.5 MB)
    target_bytes = int(target_mb * 1024 * 1024)
    
    print(f"🎯 Target File Size: ~{target_mb} MB ({target_bytes:,} bytes)")
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".tiff", ".tif")):
            continue
            
        file_path = os.path.join(folder_path, filename)
        original_size = os.path.getsize(file_path)
        
        # If it's already safe (e.g. 5MB or 9MB), skip it
        if original_size < (10 * 1024 * 1024):
            print(f"✅ Skipped (Already safe size): {filename} ({original_size/1024/1024:.2f} MB)")
            continue

        print(f"📉 Resizing: {filename} ({original_size/1024/1024:.2f} MB)...")

        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                
                # --- CALCULATION MAGIC ---
                # Uncompressed TIFF size is roughly: Width * Height * 3 bytes (for RGB)
                # We want: Width_new * Height_new * 3 ≈ Target_Bytes
                
                current_pixels = img.width * img.height
                # Estimated bytes per pixel (usually 3 for RGB)
                estimated_size = current_pixels * 3 
                
                # Calculate the scaling factor needed to hit the target byte count
                # Ratio of Area
                area_ratio = target_bytes / estimated_size
                
                # Ratio of Width/Height (Square root of area ratio)
                dimension_ratio = math.sqrt(area_ratio)
                
                # Apply a tiny safety buffer (0.98) to ensure we don't accidentally go over due to headers
                dimension_ratio *= 0.98
                
                new_width = int(img.width * dimension_ratio)
                new_height = int(img.height * dimension_ratio)
                
                # Resize using High Quality LANCZOS filter
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # SAVE WITHOUT COMPRESSION
                # compression=None ensures the file stays "heavy" and high quality
                img_resized.save(file_path, compression=None, dpi=img.info.get('dpi', (300, 300)))
                
                final_size = os.path.getsize(file_path)
                print(f"   ✨ Result: {final_size/1024/1024:.2f} MB")

        except Exception as e:
            print(f"   ❌ Error: {e}")

# 📂 Replace with your folder path
folder_path = r"D:\Masrafe\Masrafe extra\reasearch\latent_feature\mol_cel\figure"
fit_tiff_to_size(folder_path)