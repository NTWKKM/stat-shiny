import sys
import os
from pathlib import Path
import re

# Add the project root to sys.path to import modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from tabs._styling import get_shiny_css
except ImportError as e:
    print(f"Error: Could not import get_shiny_css from tabs._styling. {e}")
    sys.exit(1)

def generate_static_css():
    """
    Extracts CSS from the Shiny styling module and saves it to a static file.
    """
    print("Generating static CSS from tabs._styling.py...")
    
    # Get the raw CSS string (which includes <style> tags)
    raw_css = get_shiny_css()
    
    # Use regex to extract content between <style> and </style>
    # Note: re.DOTALL is used to match across multiple lines
    match = re.search(r'<style>(.*?)</style>', raw_css, re.DOTALL)
    
    if not match:
        print("Error: Could not find <style> tags in the output of get_shiny_css().")
        sys.exit(1)
        
    css_content = match.group(1).strip()
    
    # Define output path
    output_dir = project_root / "static"
    output_path = output_dir / "styles.css"
    
    # Create static directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write to styles.css
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(css_content)
    
    print(f"✅ Success! Static CSS saved to: {output_path}")

if __name__ == "__main__":
    generate_static_css()
