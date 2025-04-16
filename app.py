import streamlit as st
from dotenv import load_dotenv
import os
import fitz
from PIL import Image, ImageEnhance
import io
import pytesseract
from pydantic import BaseModel, Field
import google.generativeai as genai
import instructor
import pandas as pd
from datetime import datetime
import json
import logging
from typing import Optional, List

# Set up logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('siding_takeoff.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set page layout to wide
st.set_page_config(layout="wide")
logger.info("Set Streamlit page layout to wide")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found in .env file.")
    logger.error("Gemini API key not found in .env file.")
    st.stop()

# Configure Gemini API
logger.info("Configuring Gemini API")
genai.configure(api_key=GEMINI_API_KEY)

# Increase Pillow's image size limit
Image.MAX_IMAGE_PIXELS = 150000000
logger.info("Set Pillow MAX_IMAGE_PIXELS to 150,000,000")

# Custom create function for Gemini
def gemini_create(model_name, messages, **kwargs):
    logger.info(f"Calling Gemini with model: {model_name}")
    model = genai.GenerativeModel(model_name)
    prompt = messages[0]["content"]
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    logger.info("Received Gemini response")
    return response

# Patch Instructor with custom Gemini create function
logger.info("Patching Instructor with custom Gemini create function")
client = instructor.patch(create=gemini_create)

# Pydantic models
class PlanElement(BaseModel):
    scale: str = Field(..., description="Scale of the plan, e.g., 1/4\" = 1'-0\"")
    confidence: float = Field(..., ge=0.0, le=1.0)

class MaterialTakeOff(BaseModel):
    id: str = Field(..., description="Unique identifier, e.g., ROOF_001")
    location: str
    material: str
    quantity: Optional[float] = Field(default=0.0, description="Quantity of material")
    unit: str = Field(default="unknown", description="Unit of measurement, e.g., sq ft, linear ft")
    unit_cost: float = Field(default=0.0, description="Cost per unit in USD")
    total_cost: float = Field(default=0.0, description="Total cost in USD")
    confidence: float = Field(..., ge=0.0, le=1.0)
    coordinates: Optional[List[float]] = Field(default=None, description="Bounding box [x1, y1, x2, y2] in relative coordinates")

# Default unit costs (North Seattle estimates)
UNIT_COSTS = {
    "Vinyl Siding": 3.50,
    "Aluminum Trim": 2.00,
    "Soffit Panels": 4.00,
    "Cedar Beams": 50.00,
    "Cedar Posts": 75.00,
    "Seamed Metal Roofing": 10.00,
    "Teak Siding": 8.00,
    "Stone": 15.00,
    "Cedar Deck Rim": 5.00,
    "Cedar Deck Rim Shadow Board": 5.00,
    "Teak Railing": 12.00,
    "Concrete Cap": 20.00,
    "Water Cap Molding": 2.50,
    "Masonry Block & Stone": 15.00,
    "Cedar Deck": 6.00
}
logger.info("Loaded unit costs: %s", UNIT_COSTS)

# CSS for table styling
st.markdown("""
<style>
.stTable {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
}
.stTable th {
    background-color: #f0f2f6;
    font-weight: bold;
    padding: 10px;
    border: 1px solid #ddd;
    text-align: left;
}
.stTable td {
    padding: 10px;
    border: 1px solid #ddd;
}
.stTable tr:nth-child(even) {
    background-color: #f9f9f9;
}
.stTable tr:hover {
    background-color: #f1f1f1;
}
.summary-table td {
    text-align: center;
}
.warning-expander {
    background-color: #fff3cd;
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

st.title("Siding Quantity Take-Off Generator")
st.write("Upload a single-page PDF plan to generate a quantity take-off report.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    logger.info("PDF file uploaded: %s", uploaded_file.name)

    # Create a form for processing
    with st.form(key="process_pdf_form"):
        submit_button = st.form_submit_button(label="Process PDF")

        if submit_button:
            logger.info("Starting PDF processing")
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            if len(pdf_document) > 1:
                st.error("Please upload a single-page PDF.")
                logger.error("Multi-page PDF uploaded")
                st.stop()
            else:
                logger.info("Processing single-page PDF")
                page = pdf_document[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                logger.info("Generated image with size %dx%d", pix.width, pix.height)
                img = ImageEnhance.Contrast(img).enhance(2.0)
                st.image(img, caption="Uploaded Plan Preview", use_container_width=True)
                logger.info("Displayed image preview")

                # OCR processing
                custom_config = r'--oem 3 --psm 6'
                ocr_text = pytesseract.image_to_string(img, config=custom_config)
                
                # Analyze with Gemini
                prompt = f"""
                Extract the scale and material take-off details from this OCR text from an architectural plan:
                '{ocr_text}'
                Focus on identifying:
                - Scale (e.g., "1/4\"=1'" or similar).
                - Materials related to siding, trim, beams, posts, soffits, roofing, decking, railings, or caps (e.g., "Vinyl Siding", "Cedar Beams", "Seamed Metal Roofing", "Concrete Cap").
                - Quantities and units (e.g., "1200 sq ft", "10 each", "150 linear ft") from context, even if noisy (e.g., "21\" X 5.25\"" as linear ft).
                Provide confidence scores for each extracted item (0.0 to 1.0).
                Return a valid JSON array (not nested objects) combining both types, matching these Pydantic models:
                - PlanElement: {{scale: str, confidence: float}}
                - MaterialTakeOff: {{id: str, location: str, material: str, quantity: float, unit: str, unit_cost: float, total_cost: float, confidence: float, coordinates: list[float]}}
                For MaterialTakeOff:
                - Generate a unique 'id' (e.g., "ROOF_001", "WALL_002") based on material and location.
                - Infer 'location' from context (e.g., "Roof", "Deck", "Exterior Walls").
                - Ensure 'quantity' is a float (calculate or estimate; use 0.0 only if no data).
                - Ensure 'unit' is a valid string (e.g., "sq ft" for siding, "linear ft" for railings, "each" for beams).
                - Set 'unit_cost' to 0.0; calculate 'total_cost' as quantity * unit_cost.
                - Provide 'coordinates' as a bounding box [x1, y1, x2, y2] in relative image coordinates (0 to 1, where image is 10800x7200 pixels):
                  - Roof: Top 30% (e.g., [0.2, 0.0, 0.8, 0.3]).
                  - Walls: Middle 40% (e.g., [0.2, 0.3, 0.8, 0.7]).
                  - Deck: Bottom 20% (e.g., [0.3, 0.7, 0.7, 0.9]).
                  - Railings/Molding: Edges (e.g., [0.2, 0.7, 0.8, 0.75]).
                  - Adjust based on OCR context (e.g., "Ridge" at top).
                Calculations:
                - Walls (e.g., "Teak Siding"):
                  - Use height (e.g., "10.1'") and estimate perimeter (e.g., 50' total) or extract width.
                  - Area = height * width (e.g., 10.1' * 50' = 505 sq ft).
                  - Coordinates: Middle section.
                - Roofs (e.g., "Seamed Metal Roofing"):
                  - Use ridge height ("16.5'") and estimate base (e.g., 20' width).
                  - Area = length * width, adjust for slope if noted (e.g., 16.5' * 20' = 330 sq ft, or use "125" if explicit).
                  - Coordinates: Top section.
                - Linear items (e.g., "Water Cap Molding"):
                  - Use perimeter (e.g., 50' for walls) or explicit lengths ("6\" CONCRETE CAP" = 6 linear ft).
                  - Coordinates: Edge strip.
                - Counts (e.g., "Cedar Beams & Posts"):
                  - Estimate from context (e.g., assume 4 posts for "4.25\" X 4.25\"").
                  - Coordinates: Small boxes or points.
                - Decks (e.g., "Cedar Deck"):
                  - Estimate area (e.g., 20' * 10' = 200 sq ft if no dimensions).
                  - Coordinates: Lower section.
                Use scale (1/4"=1') to convert dimensions (1 inch = 4 feet).
                Log calculations (e.g., "Teak Siding: 10.1' * 50' = 505 sq ft").
                Ensure no null values; include all materials like "Concrete Cap", "Cedar Deck Rim Shadow Board".
                Output a flat JSON array, e.g., [{{"scale": "1/4\"=1'", "confidence": 0.95}}, {{"id": "ROOF_001", "location": "Roof", "material": "Seamed Metal Roofing", "quantity": 125.0, "unit": "sq ft", "unit_cost": 0.0, "total_cost": 0.0, "confidence": 0.8, "coordinates": [0.2, 0.0, 0.8, 0.3]}}].
                """
                logger.info("Sending prompt to Gemini")
                response = client(
                    model_name="gemini-1.5-flash",
                    messages=[{"role": "user", "content": prompt}]
                )

                # Display extracted text and raw response in a closed expander
                with st.expander("Debug Information", expanded=False):
                    st.subheader("Extracted Text")
                    st.text_area("OCR Output", ocr_text, height=200)
                    logger.info("Displayed extracted text in expander")
                    
                    st.subheader("Raw Gemini Response")
                    st.text(response.text)
                    logger.info("Displayed raw Gemini response in expander")

                # Manually parse the JSON response
                try:
                    response_data = json.loads(response.text)
                    logger.info("Parsed JSON response: %s", response_data)
                    parsed_response = []
                    warnings = []
                    for item in response_data:
                        logger.debug("Processing item: %s", item)
                        try:
                            parsed_item = PlanElement(**item) if "scale" in item else MaterialTakeOff(**item)
                            parsed_response.append(parsed_item)
                        except Exception as e:
                            logger.error("Validation error for item %s: %s", item, e)
                            warnings.append(f"Skipped invalid item {item.get('material', 'unknown')}: {str(e)}")
                    logger.info("Parsed %d items into Pydantic models", len(parsed_response))
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse Gemini response as JSON: {e}")
                    logger.error("JSON parsing error: %s", e)
                    st.text(response.text)
                    parsed_response = []
                    warnings = ["Failed to parse Gemini response"]

                # Process response into report
                logger.info("Generating report")
                summary = {
                    "Date": datetime.now().strftime("%m/%d/%Y"),
                    "Project": "Unknown",
                    "Page Number": 1,
                    "Number of Floors": "Unknown",
                    "Description": "Extracted from PDF"
                }
                takeoff_data = []
                cost_data = []

                for item in parsed_response:
                    logger.debug("Processing report item: %s", item)
                    if isinstance(item, PlanElement):
                        summary["Description"] += f" (Scale: {item.scale}, Confidence: {item.confidence*100:.0f}%)"
                        if item.confidence < 0.7:
                            warnings.append(f"Low confidence ({item.confidence*100:.0f}%) for scale: {item.scale}")
                    elif isinstance(item, MaterialTakeOff):
                        if item.unit_cost == 0.0 and item.material in UNIT_COSTS:
                            item.unit_cost = UNIT_COSTS[item.material]
                        item.total_cost = item.quantity * item.unit_cost
                        takeoff_data.append([item.id, item.location, item.material, item.quantity, item.unit, f"{item.confidence*100:.0f}%"])
                        cost_data.append([item.id, item.material, item.quantity, f"${item.unit_cost:.2f}", f"${item.total_cost:.2f}", f"{item.confidence*100:.0f}%"])
                        if item.confidence < 0.7:
                            warnings.append(f"Low confidence ({item.confidence*100:.0f}%) for {item.material} at {item.location}")
                        if item.unit == "unknown":
                            warnings.append(f"Unit inferred as 'unknown' for {item.material} at {item.location}; please verify")
                        if item.quantity == 0.0:
                            warnings.append(f"No quantity extracted for {item.material} at {item.location}; please verify")
                    logger.info("Processed item: %s", item)

                # Display report within form
                st.subheader("Take-Off Report")
                # Summary
                st.markdown("### Summary")
                summary_df = pd.DataFrame([summary])
                st.markdown(
                    summary_df.to_html(classes="summary-table stTable", index=False),
                    unsafe_allow_html=True
                )
                logger.info("Displayed summary table")

                # Quantity Take-Off
                st.markdown("### Quantity Take-Off")
                takeoff_df = pd.DataFrame(
                    takeoff_data,
                    columns=["ID", "Location", "Material", "Quantity", "Unit of Measurement", "Confidence Score"]
                )
                st.markdown(
                    takeoff_df.to_html(classes="stTable", index=False, escape=False),
                    unsafe_allow_html=True
                )
                logger.info("Displayed quantity take-off table")

                # Cost Estimation
                st.markdown("### Cost Estimation")
                cost_df = pd.DataFrame(
                    cost_data,
                    columns=["ID", "Material", "Quantity", "Unit Cost", "Total Cost", "Confidence Score"]
                )
                st.markdown(
                    cost_df.to_html(classes="stTable", index=False, escape=False),
                    unsafe_allow_html=True
                )
                logger.info("Displayed cost estimation table")

                # Warnings
                if warnings:
                    st.markdown("### Notes/Warnings")
                    with st.expander("View Warnings", expanded=True):
                        for warning in warnings:
                            st.markdown(f"- {warning}", unsafe_allow_html=True)
                    logger.info("Displayed warnings: %s", warnings)