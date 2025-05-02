import numpy as np
from PIL import Image, ImageDraw
import os

def create_dashboard_collage():
    # Define the visualization files to include
    vis_files = [
        'correlation_heatmap.png',
        'categorical_vs_exam_score.png',
        'numerical_vs_exam_score.png',
        'categorical_distributions.png',
        'numerical_distributions.png'
    ]
    
    # Load images
    images = []
    for file in vis_files:
        try:
            img_path = os.path.join('visualizations', file)
            img = Image.open(img_path)
            # Convert to RGBA if not already
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            # Resize to a consistent width while maintaining aspect ratio
            basewidth = 1000
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
            images.append(img)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    if not images:
        print("No images could be loaded!")
        return
    
    # Calculate dimensions for the collage
    padding = 40
    margin = 60
    title_height = 100
    
    # Create a 2x3 grid
    n_rows = 2
    n_cols = 3
    
    cell_width = 1000 + padding * 2
    cell_height = 700 + padding * 2
    
    collage_width = cell_width * n_cols + margin * 2
    collage_height = cell_height * n_rows + margin * 2 + title_height
    
    # Create new image with dark background
    background_color = (8, 15, 40)  # Dark blue
    collage = Image.new('RGBA', (collage_width, collage_height), background_color)
    draw = ImageDraw.Draw(collage)
    
    # Paste images into the collage
    for idx, img in enumerate(images):
        if idx >= n_rows * n_cols:
            break
            
        row = idx // n_cols
        col = idx % n_cols
        
        # Calculate position
        x = col * cell_width + margin + padding
        y = row * cell_height + margin + padding + title_height
        
        # Create a mask for rounded corners
        mask = Image.new('L', img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle([0, 0, img.size[0], img.size[1]], radius=20, fill=255)
        
        # Add a subtle glow effect
        glow = Image.new('RGBA', img.size, (255, 255, 255, 0))
        glow_draw = ImageDraw.Draw(glow)
        glow_draw.rounded_rectangle([0, 0, img.size[0], img.size[1]], radius=20, 
                                  fill=(255, 255, 255, 30))
        collage.paste(glow, (x-5, y-5), glow)
        
        # Paste the image
        collage.paste(img, (x, y), mask)
    
    # Save the collage
    os.makedirs('images', exist_ok=True)
    collage_path = os.path.join('images', 'dashboard.png')
    collage.save(collage_path, quality=95)
    print(f"Collage saved as {collage_path}")

if __name__ == "__main__":
    create_dashboard_collage() 