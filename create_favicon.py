#!/usr/bin/env python3
"""
Simple script to create a favicon for the Loan Prediction System
Creates a simple 32x32 pixel ICO file
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    import os

    def create_favicon():
        # Create a 32x32 image with a blue background
        size = (32, 32)
        img = Image.new('RGBA', size, (45, 114, 214, 255))  # Primary blue color
        draw = ImageDraw.Draw(img)

        # Draw a simple bank/document icon
        # White rectangle (document)
        draw.rectangle([6, 8, 26, 24], fill=(255, 255, 255, 255))

        # Blue lines (text lines)
        draw.rectangle([8, 11, 16, 12], fill=(45, 114, 214, 255))
        draw.rectangle([8, 14, 20, 15], fill=(45, 114, 214, 255))
        draw.rectangle([8, 17, 18, 18], fill=(45, 114, 214, 255))

        # Green checkmark/approval symbol
        draw.ellipse([18, 13, 24, 19], fill=(16, 185, 129, 255))
        draw.line([(19, 16), (21, 17.5), (23, 14.5)], fill=(255, 255, 255, 255), width=2)

        # Save as ICO file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        frontend_dir = os.path.join(script_dir, 'frontend')

        # Create frontend directory if it doesn't exist
        os.makedirs(frontend_dir, exist_ok=True)

        favicon_path = os.path.join(frontend_dir, 'favicon.ico')
        img.save(favicon_path, format='ICO', sizes=[size])

        print(f"✅ Favicon created successfully at: {favicon_path}")
        return True

    if __name__ == "__main__":
        create_favicon()

except ImportError:
    print("❌ Pillow (PIL) not installed. Installing...")
    import subprocess
    import sys

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        print("✅ Pillow installed successfully. Please run this script again.")
    except subprocess.CalledProcessError:
        print("❌ Failed to install Pillow. Creating simple text favicon instead...")

        # Create a simple text-based favicon fallback
        script_dir = os.path.dirname(os.path.abspath(__file__))
        frontend_dir = os.path.join(script_dir, 'frontend')
        os.makedirs(frontend_dir, exist_ok=True)
        favicon_path = os.path.join(frontend_dir, 'favicon.ico')

        # Create minimal ICO file header (this is a very basic approach)
        # For production, you'd want to use proper ICO format
        with open(favicon_path, 'wb') as f:
            # Write minimal ICO header
            f.write(b'\x00\x00\x01\x00\x01\x00\x20\x20\x00\x00\x01\x00\x20\x00')
            f.write(b'\x00\x00\x00\x00\x16\x00\x00\x00')
            # Write minimal bitmap data (32x32 blue square)
            for _ in range(32 * 32):
                f.write(b'\x2d\x72\xd6\xff')  # Blue color in BGRA format

        print(f"✅ Simple favicon created at: {favicon_path}")