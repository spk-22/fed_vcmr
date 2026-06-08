"""
Convert TikZ to PNG using online rendering service or local imagemagick

Alternative: Generate clean matplotlib/plotly version with professional styling
"""

import subprocess
import os

# Check the .tex file
tex_file = "C:/prism/outputs/figures/fedvcmr_architecture.tex"
print(f"✓ TikZ source created: {tex_file}")
print(f"\n📋 OPTIONS TO RENDER:")
print(f"\n1️⃣  ONLINE (No software needed):")
print(f"   - Go to: https://www.overleaf.com")
print(f"   - Create new project → Upload from file")
print(f"   - Upload: {tex_file}")
print(f"   - Download as PDF/PNG")

print(f"\n2️⃣  LOCAL (Windows):")
print(f"   - Install MiKTeX: choco install miktex")
print(f"   - Then run: pdflatex {tex_file}")

print(f"\n3️⃣  DOCKER (if you have Docker):")
print(f'   - docker run --rm -v "C:/prism/outputs/figures:/workspace" texlive/texlive pdflatex -output-directory=/workspace fedvcmr_architecture.tex')

# Try to find ImageMagick or other converters
try:
    result = subprocess.run(['convert', '--version'], capture_output=True, text=True, timeout=2)
    if result.returncode == 0:
        print(f"\n✓ ImageMagick found! Can convert PDF→PNG once compiled")
except:
    print(f"\n   To enable PNG conversion later: choco install imagemagick")

print(f"\n{'='*60}")
print(f"📄 TikZ file ready for compilation:")
print(f"   {tex_file}")
print(f"\n💡 TikZ advantages:")
print(f"   ✓ Publication-quality (journal-ready)")
print(f"   ✓ Perfect for ML/research papers")
print(f"   ✓ Version control friendly (pure text)")
print(f"   ✓ Extremely clean output")
