from pptx import Presentation
import os

SOURCE_PPTX = "/Users/ashishb/Documents/cap/NYC-Citi-Bike-Demand-A-Big-Data-Exploratory-Analysis-and-Advanced-Interactive-Dashboard.pptx"

if os.path.exists(SOURCE_PPTX):
    prs = Presentation(SOURCE_PPTX)
    print(f"Loaded template from {SOURCE_PPTX}")
    
    print(f"Number of layouts: {len(prs.slide_layouts)}")
    
    for i, layout in enumerate(prs.slide_layouts):
        print(f"\nLayout {i}: {layout.name}")
        for shape in layout.placeholders:
            print(f"  Placeholder idx {shape.placeholder_format.idx}: type {shape.placeholder_format.type} name '{shape.name}'")
