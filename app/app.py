"""
SatQuery AI — Interactive Streamlit Web GUI.
Supports Single Image (VQA, Captioning, Grounding),
Cross-Modal Pair (Optical-SAR Fusion), and Bi-temporal Change Detection.

Run locally:
    streamlit run app/app.py
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

from src.utils.io_utils import load_config, save_uploaded_file
from src.preprocessing.geotiff_utils import to_pil_rgb
from src.agent.controller import AgentController


st.set_page_config(
    page_title="SatQuery AI — Remote Sensing Assistant",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

config = load_config("configs/config.yaml")


@st.cache_resource
def get_controller():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AgentController(config, device=device), device


controller, device_used = get_controller()

# Header
st.title("🛰️ SatQuery AI")
st.markdown(
    f"**Autonomous Vision-Language & Multi-Modal Satellite Intelligence Agent** "
    f"*(Running on: `{device_used.upper()}`)*"
)
st.caption(
    "Query satellite imagery using natural language: VQA, Scene Captioning, "
    "Spatial Object Grounding, and Optical + SAR Radar Fusion."
)

# Preset Samples
PRESET_SINGLE_IMAGES = {
    "VRSBench: Airport with Yellow Buses": {
        "path": "data/P0003_0002.png",
        "description": "High-resolution optical scene containing an airport apron and yellow transport buses.",
        "default_query": "What color are the buses in the image?",
        "suggested_queries": [
            "What color are the buses in the image?",
            "Are there buses parked in the parking lot?",
            "How many yellow buses are visible?",
            "Describe the overall scene and layout.",
            "Localize the yellow bus in the scene.",
        ],
    },
    "VRSBench: Airport with Airplane on Tarmac": {
        "path": "data/vrsbench_samples/airport_airplanes.png",
        "description": "High-resolution optical aerial scene of an active airport runway, taxiway, and airplane.",
        "default_query": "Which object is visible on the tarmac?",
        "suggested_queries": [
            "Which object is visible on the tarmac?",
            "Is the airplane positioned near taxiway intersections?",
            "Are there any grassy areas visible in the image?",
            "Describe this airport scene in detail.",
            "Localize the airplane on the tarmac.",
            "Mark the grass in this image.",
        ],
    },
    "VRSBench: Harbor & Ships on Water": {
        "path": "data/vrsbench_samples/harbor_ships.png",
        "description": "High-resolution optical imagery showing coastal waterways, piers, and ships docked in harbor.",
        "default_query": "What is the primary natural feature shown in the image?",
        "suggested_queries": [
            "What is the primary natural feature shown in the image?",
            "How many harbors are visible in the image?",
            "Is the small ship closer to the top or bottom of the image?",
            "Describe the marine harbor and surrounding area.",
            "Localize the ship in the water.",
        ],
    },
    "VRSBench: Industrial Circular Storage Tanks": {
        "path": "data/vrsbench_samples/storage_tanks.png",
        "description": "High-resolution industrial infrastructure showing circular storage containers and facilities.",
        "default_query": "How many storage tanks are in the image?",
        "suggested_queries": [
            "How many storage tanks are in the image?",
            "What shape does the storage tank appear to have in the image?",
            "Where is the storage tank located within the image?",
            "Describe this industrial scene.",
            "Localize the storage tank.",
        ],
    },
    "VRSBench: Sports Stadium & Parking Lot": {
        "path": "data/vrsbench_samples/sports_stadium.png",
        "description": "High-resolution optical view of a large sports arena with seating rows and adjacent parking lot.",
        "default_query": "What structures are adjacent to the large stadium?",
        "suggested_queries": [
            "What structures are adjacent to the large stadium?",
            "How many small vehicles can be distinctly identified in the parking lot?",
            "Where is the parking lot in relation to the stadium?",
            "Describe the overall stadium complex.",
            "Localize the parking lot vehicles.",
        ],
    },
    "BigEarthNet: Dense Forest Scene": {
        "path": "data/sample_bigearthnet/forest_s2_rgb.png",
        "description": "Sentinel-2 optical RGB composite showing mixed and coniferous tree canopies.",
        "default_query": "Is there a forest in this satellite image?",
        "suggested_queries": [
            "Is there a forest in this satellite image?",
            "Are there trees in this area?",
            "Describe the natural vegetation in this image.",
            "Localize the forest canopy.",
        ],
    },
    "BigEarthNet: Coastal Waters & Ocean": {
        "path": "data/sample_bigearthnet/water_coastal_s2_rgb.png",
        "description": "Sentinel-2 optical RGB composite depicting open water and coastline.",
        "default_query": "Is there water or ocean in this image?",
        "suggested_queries": [
            "Is there water or ocean in this image?",
            "What type of natural water body is visible?",
            "Describe the shoreline and water features.",
            "Localize the water body.",
        ],
    },
    "BigEarthNet: Urban & Built-up Fabric": {
        "path": "data/sample_bigearthnet/urban_s2_rgb.png",
        "description": "Sentinel-2 optical RGB composite of developed human settlement and buildings.",
        "default_query": "Is this an urban area or open forest?",
        "suggested_queries": [
            "Is this an urban area or open forest?",
            "Are there buildings or human structures here?",
            "Describe the built-up area and infrastructure.",
            "Localize the urban developed region.",
        ],
    },
}

PRESET_CROSS_MODAL = {
    "BigEarthNet: Co-registered Sentinel-1 SAR + Sentinel-2 Optical (GeoTIFF)": {
        "optical": "data/sample_bigearthnet/sample_s2.tif",
        "sar": "data/sample_bigearthnet/sample_s1.tif",
        "optical_preview": "data/sample_bigearthnet/forest_s2_rgb.png",
        "sar_preview": "data/sample_bigearthnet/forest_s1_sar.png",
        "description": "Raw BigEarthNet-MM GeoTIFF pair: 12-band Sentinel-2 reflectance + dual-pol Sentinel-1 SAR backscatter.",
        "default_query": "Analyze this multi-modal pair and classify the land cover.",
    },
    "BigEarthNet: Dense Woodland & Forest Canopy (Optical + SAR)": {
        "optical": "data/sample_bigearthnet/forest_s2_rgb.png",
        "sar": "data/sample_bigearthnet/forest_s1_sar.png",
        "optical_preview": "data/sample_bigearthnet/forest_s2_rgb.png",
        "sar_preview": "data/sample_bigearthnet/forest_s1_sar.png",
        "description": "Woodland canopy showing strong NIR reflectance in optical and diffuse volume scattering in radar.",
        "default_query": "Analyze this multi-modal pair and identify the forest cover.",
    },
    "BigEarthNet: Urban Fabric & Industrial Built-up (Optical + SAR)": {
        "optical": "data/sample_bigearthnet/urban_s2_rgb.png",
        "sar": "data/sample_bigearthnet/urban_s1_sar.png",
        "optical_preview": "data/sample_bigearthnet/urban_s2_rgb.png",
        "sar_preview": "data/sample_bigearthnet/urban_s1_sar.png",
        "description": "Developed settlement: concrete/metallic buildings produce intense double-bounce radar flares.",
        "default_query": "Analyze this multi-modal pair and classify the urban structures.",
    },
    "BigEarthNet: Agricultural Cropland & Fields (Optical + SAR)": {
        "optical": "data/sample_bigearthnet/agriculture_s2_rgb.png",
        "sar": "data/sample_bigearthnet/agriculture_s1_sar.png",
        "optical_preview": "data/sample_bigearthnet/agriculture_s2_rgb.png",
        "sar_preview": "data/sample_bigearthnet/agriculture_s1_sar.png",
        "description": "Cropland parcels displaying soil moisture radar backscatter and distinct crop field geometry.",
        "default_query": "Analyze this multi-modal pair and assess the agricultural land.",
    },
    "BigEarthNet: Coastal Shoreline & Water Body (Optical + SAR)": {
        "optical": "data/sample_bigearthnet/water_coastal_s2_rgb.png",
        "sar": "data/sample_bigearthnet/water_coastal_s1_sar.png",
        "optical_preview": "data/sample_bigearthnet/water_coastal_s2_rgb.png",
        "sar_preview": "data/sample_bigearthnet/water_coastal_s1_sar.png",
        "description": "Coastal waters exhibiting mirror-like specular radar reflection (appearing pitch dark).",
        "default_query": "Analyze this multi-modal pair and identify water bodies and coastlines.",
    },
}

PRESET_BI_TEMPORAL = {
    "Bi-Temporal: Forest Clearing & New Industrial Facility": {
        "image_t1": "data/sample_bitemporal/bitemporal_t1_before.png",
        "image_t2": "data/sample_bitemporal/bitemporal_t2_after.png",
        "description": "Co-registered bi-temporal pair showing pristine forest with river and road (T1) vs forest clearing with newly constructed industrial facility and access road (T2).",
        "default_query": "What changed between time T1 and time T2?",
        "suggested_queries": [
            "What changed between time T1 and time T2?",
            "Detect any new construction or deforestation between T1 and T2.",
            "Where did the changes occur between the two dates?",
            "Show the difference mask between time T1 and time T2.",
        ],
    }
}


def draw_bounding_box(image_path: str, bbox: list = None, bboxes: list = None, label: str = "GROUNDING") -> Image.Image:
    try:
        from PIL import ImageDraw
        from src.preprocessing.geotiff_utils import to_pil_rgb
        img = to_pil_rgb(image_path)
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        box_list = []
        if bboxes and len(bboxes) > 0:
            box_list = bboxes
        elif bbox and len(bbox) == 4:
            box_list = [bbox]
            
        for b in box_list:
            x1, y1, x2, y2 = b
            # Clamp to bounds
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            
            draw.rectangle([x1, y1, x2, y2], outline="#00FF66", width=4)
            badge_w = min(w - x1, max(60, len(label) * 9 + 12))
            draw.rectangle([x1, y1, x1 + badge_w, y1 + 22], fill="#00FF66")
            draw.text((x1 + 6, y1 + 4), label[:18], fill="#000000")
        return img
    except Exception as e:
        st.warning(f"Could not render bounding box overlay: {e}")
        return None


# Sidebar
st.sidebar.header("1. Input Configuration")
input_mode = st.sidebar.radio(
    "Select Modality / Analysis Mode",
    ["Single Image (VQA / Grounding / Captioning)", "Optical + SAR Pair (Fusion)", "Bi-Temporal Pair (Change Detection)"],
)

image_inputs = {}
selected_preset_key = None
active_sample_info = None

if input_mode.startswith("Single"):
    source_choice = st.sidebar.radio("Image Source", ["Use Preloaded Sample", "Upload Custom File"])
    if source_choice == "Use Preloaded Sample":
        selected_preset_key = st.sidebar.selectbox("Choose a benchmark satellite image:", list(PRESET_SINGLE_IMAGES.keys()))
        active_sample_info = PRESET_SINGLE_IMAGES[selected_preset_key]
        image_inputs["image"] = active_sample_info["path"]
        st.sidebar.info(active_sample_info["description"])
    else:
        uploaded_file = st.sidebar.file_uploader("Upload satellite image (PNG/JPEG/TIFF):", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded_file:
            image_inputs["image"] = save_uploaded_file(uploaded_file, target_dir="data/raw/uploads")

elif input_mode.startswith("Optical + SAR"):
    source_choice = st.sidebar.radio("Pair Source", ["Use Preloaded Multi-Modal Pair", "Upload Custom Pair"])
    if source_choice == "Use Preloaded Multi-Modal Pair":
        selected_preset_key = st.sidebar.selectbox("Choose a multi-modal pair:", list(PRESET_CROSS_MODAL.keys()))
        active_sample_info = PRESET_CROSS_MODAL[selected_preset_key]
        image_inputs["image_optical"] = active_sample_info["optical"]
        image_inputs["image_sar"] = active_sample_info["sar"]
        st.sidebar.info(active_sample_info["description"])
    else:
        up_opt = st.sidebar.file_uploader("Upload Optical Image (GeoTIFF/PNG):", type=["tif", "tiff", "png", "jpg", "jpeg"], key="opt_up")
        up_sar = st.sidebar.file_uploader("Upload SAR Image (GeoTIFF/PNG):", type=["tif", "tiff", "png", "jpg", "jpeg"], key="sar_up")
        if up_opt and up_sar:
            image_inputs["image_optical"] = save_uploaded_file(up_opt, target_dir="data/raw/uploads")
            image_inputs["image_sar"] = save_uploaded_file(up_sar, target_dir="data/raw/uploads")

else:
    source_choice = st.sidebar.radio("Pair Source", ["Use Preloaded Bi-Temporal Pair", "Upload Custom Pair"])
    if source_choice == "Use Preloaded Bi-Temporal Pair":
        selected_preset_key = list(PRESET_BI_TEMPORAL.keys())[0]
        active_sample_info = PRESET_BI_TEMPORAL[selected_preset_key]
        image_inputs["image_t1"] = active_sample_info["image_t1"]
        image_inputs["image_t2"] = active_sample_info["image_t2"]
        st.sidebar.info(active_sample_info["description"])
    else:
        st.sidebar.markdown("Upload two co-registered images taken at different times (T1 and T2).")
        up_t1 = st.sidebar.file_uploader("Image T1 (Before):", type=["tif", "tiff", "png", "jpg", "jpeg"], key="t1_up")
        up_t2 = st.sidebar.file_uploader("Image T2 (After):", type=["tif", "tiff", "png", "jpg", "jpeg"], key="t2_up")
        if up_t1 and up_t2:
            image_inputs["image_t1"] = save_uploaded_file(up_t1, target_dir="data/raw/uploads")
            image_inputs["image_t2"] = save_uploaded_file(up_t2, target_dir="data/raw/uploads")

# Sidebar: Query input
st.sidebar.header("2. Ask Your Question")

default_q = ""
if active_sample_info and "default_query" in active_sample_info:
    default_q = active_sample_info["default_query"]
elif input_mode.startswith("Optical + SAR"):
    default_q = "Analyze this multi-modal pair and classify the land cover."
elif input_mode.startswith("Bi-Temporal"):
    default_q = "What changed between time T1 and time T2?"
else:
    default_q = "Describe what is shown in this satellite image."

if active_sample_info and "suggested_queries" in active_sample_info:
    with st.sidebar.expander("💡 Clickable Query Suggestions", expanded=True):
        for sq in active_sample_info["suggested_queries"]:
            if st.button(sq, key=f"btn_{sq[:15]}"):
                st.session_state["active_query"] = sq

query_val = st.session_state.get("active_query", default_q)
query_text = st.sidebar.text_area("Your Question / Instruction:", value=query_val, height=90)

run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main Screen
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("🖼️ Input Satellite Imagery")
    if input_mode.startswith("Single") and "image" in image_inputs:
        img_source = image_inputs["image"]
        is_sar = False
        if isinstance(img_source, str) and ("s1" in Path(img_source).name.lower() or "grd" in Path(img_source).name.lower()):
            is_sar = True

        sar_mode = "grayscale"
        if is_sar:
            sar_choice = st.radio(
                "📡 Radar Visualization Mode:",
                ["High-Contrast Calibrated Intensity (Clean B&W)", "Dual-Pol False Color (VV/VH)"],
                index=0,
                horizontal=True,
                help="Radar measures microwave roughness, not camera color. Intensity mode makes buildings and structures glow white against dark fields.",
            )
            sar_mode = "grayscale" if "Intensity" in sar_choice else "composite"

        disp_img = to_pil_rgb(img_source, sar_mode=sar_mode)
        caption_text = "Sentinel-1 SAR Radar (Enhanced)" if is_sar else "Selected Satellite Scene (Processed)"
        st.image(disp_img, caption=caption_text, use_container_width=True)

        if is_sar:
            st.markdown(
                """
                > 📡 **Understanding Sentinel-1 Radar:**  
                > Unlike standard cameras, **SAR Radar** emits microwave pulses to see through clouds and night:  
                > • ⬜ **Bright White Spots:** Man-made structures, roofs & buildings (radar corner reflection)  
                > • ⬛ **Dark / Black Parcels:** Water bodies, smooth roads, or calm flat fields  
                > • 🌫️ **Textured Mid-tones:** Forest canopies and agricultural crops  
                >  
                > 💡 **Want true-color optical photos of buildings, airports & vehicles?**  
                > Switch to **Use Preloaded Sample** in the sidebar and choose any **VRSBench** scene, or upload the matching **`S2A_...`** optical file!
                """
            )
    elif input_mode.startswith("Optical + SAR") and "image_optical" in image_inputs:
        col_opt, col_sar = st.columns(2)
        with col_opt:
            prev_opt = active_sample_info.get("optical_preview", image_inputs["image_optical"]) if active_sample_info else image_inputs["image_optical"]
            st.image(to_pil_rgb(prev_opt), caption="Sentinel-2 Optical (RGB)", use_container_width=True)
        with col_sar:
            prev_sar = active_sample_info.get("sar_preview", image_inputs["image_sar"]) if active_sample_info else image_inputs["image_sar"]
            st.image(to_pil_rgb(prev_sar), caption="Sentinel-1 SAR Radar", use_container_width=True)
    elif "image_t1" in image_inputs and "image_t2" in image_inputs:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.image(to_pil_rgb(image_inputs["image_t1"]), caption="Time T1 (Before)", use_container_width=True)
        with col_t2:
            st.image(to_pil_rgb(image_inputs["image_t2"]), caption="Time T2 (After)", use_container_width=True)
    else:
        st.info("👈 Select a sample from the sidebar or upload an image to begin.")

with col_right:
    st.subheader("🤖 Agentic Result & Evidence")
    
    if run_button:
        if not image_inputs:
            st.error("Please provide the required image input.")
        elif not query_text.strip():
            st.error("Please type a question or instruction.")
        else:
            with st.spinner("SatQuery Agent routing and running specialist model..."):
                result = controller.run(images=image_inputs, query=query_text)
            
            if not result.get("success"):
                st.error("Analysis error: " + "; ".join(result.get("errors", ["Unknown error"])))
            else:
                task_name = result.get("task", "General").upper()
                st.success(f"**Routed Specialist:** `{task_name}`")
                
                # Answer Text
                st.markdown("### Response:")
                st.markdown(f"> **{result.get('text')}**")
                
                # Evidence Verification Gate Audit Card
                verif = result.get("verification") or {}
                decision = verif.get("decision", "SUPPORTED")
                status_val = result.get("status", "unknown")

                if decision == "SUPPORTED":
                    st.success("🛡️ **Evidence Verification Gate:** `SUPPORTED` — Model prediction passed all semantic, task, and sanity verification checks.")
                elif decision == "FALLBACK_GEE":
                    st.warning("🛰️ **Evidence Verification Gate:** `FALLBACK TO GEE ACTIVATED` — Model prediction unverified; routed to Google Earth Engine catalog.")
                else:
                    st.error("⚠️ **Evidence Verification Gate:** `UNVERIFIED` — Model output flagged for potential hallucination or task mismatch.")

                # GEE Fallback Panel
                gee_ev = verif.get("gee_evidence")
                if decision == "FALLBACK_GEE" and gee_ev:
                    with st.container(border=True):
                        st.markdown("#### 🛰️ Google Earth Engine (GEE) Fallback Package")
                        st.info(gee_ev.get("action", "GEE query triggered to replace uncalibrated inference."))
                        col_g1, col_g2 = st.columns(2)
                        with col_g1:
                            st.markdown(f"**Engine:** `{gee_ev.get('engine')}`")
                            st.markdown(f"**Collection:** `{gee_ev.get('collection')}`")
                        with col_g2:
                            st.markdown(f"**Target Entity:** `{gee_ev.get('target')}`")
                            st.markdown(f"**Coordinates / Tile:** `{gee_ev.get('coordinates')}`")
                        st.caption(f"**Verification Method:** {gee_ev.get('verification_method')}")

                # Specific Gate Review Findings (if flagged)
                if verif.get("reasons"):
                    with st.container(border=True):
                        st.markdown("##### ⚠️ Verification Gate Audit Findings:")
                        for r in verif["reasons"]:
                            st.markdown(f"- {r}")

                # Multi-Criteria Audit Breakdown
                checks = verif.get("checks", {})
                with st.expander("🔬 Verification Gate Multi-Criteria Breakdown", expanded=(decision != "SUPPORTED")):
                    tc = checks.get("task_compliance", {})
                    ta = checks.get("target_alignment", {})
                    os_chk = checks.get("output_sanity", {})
                    geo_chk = checks.get("geospatial_context", {})

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**1. Task Compliance:** {'✅ PASS' if tc.get('passed') else '❌ FAIL'}")
                        st.caption(tc.get("detail", "N/A"))

                        target_label = ta.get("target") or "None detected"
                        st.markdown(f"**2. Target Semantic Alignment:** {'✅ PASS' if ta.get('passed') else '❌ FAIL'} (`{target_label}`)")
                        st.caption(ta.get("detail", "N/A"))

                    with c2:
                        st.markdown(f"**3. Output Sanity & Repetition:** {'✅ PASS' if os_chk.get('passed') else '❌ FAIL'}")
                        st.caption(os_chk.get("detail", "N/A"))

                        geo_avail = geo_chk.get("available", False)
                        st.markdown(f"**4. Geospatial Context (GEE):** {'🌐 Georeferenced' if geo_avail else '📄 Benchmark Crop'}")
                        meta_info = geo_chk.get("metadata", {})
                        if geo_avail:
                            st.caption(f"Type: {meta_info.get('type')} | Tile/CRS: {meta_info.get('tile') or meta_info.get('crs')}")
                        else:
                            st.caption(meta_info.get("reason", "No CRS or MGRS tile metadata attached."))

                # Confidence Meter & Raw Metadata
                conf = float(result.get("confidence", 0.0) or 0.0)
                st.progress(min(max(conf, 0.0), 1.0))
                st.caption(f"Raw Model Confidence: **{conf * 100:.1f}%** | Overall Gate Status: `{status_val}` (Guarded by Verification Gate)")

                
                # Spatial Evidence (Bounding Box)
                evidence = result.get("spatial_evidence") or {}
                if evidence.get("type") == "bbox" and evidence.get("bbox"):
                    bbox = evidence["bbox"]
                    meta_params = (result.get("metadata") or {}).get("parameters", {})
                    bboxes = meta_params.get("bboxes") or [bbox]
                    target_name = meta_params.get("target", "")
                    badge_label = target_name.upper() if target_name and target_name != "target" else task_name

                    if len(bboxes) > 1:
                        st.markdown(f"#### 🎯 Grounded Target Localization ({len(bboxes)} regions detected):")
                        cols_b = st.columns(min(3, len(bboxes)))
                        for idx, b in enumerate(bboxes, 1):
                            c_idx = (idx - 1) % len(cols_b)
                            with cols_b[c_idx]:
                                st.caption(f"**Region {idx}:** `[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]`")
                    else:
                        st.markdown("#### 🎯 Grounded Target Localization:")
                        st.code(f"Bounding Box Coordinates: [x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, x2={bbox[2]:.1f}, y2={bbox[3]:.1f}]")

                    if "image" in image_inputs:
                        annotated = draw_bounding_box(image_inputs["image"], bbox=bbox, bboxes=bboxes, label=badge_label)
                        if annotated:
                            box_caption = f"Grounded Evidence: {badge_label} ({len(bboxes)} Neon Boxes)" if len(bboxes) > 1 else f"Grounded Evidence: {badge_label} (Neon Box)"
                            st.image(annotated, caption=box_caption, use_container_width=True)
                            
                elif evidence.get("type") == "mask" and evidence.get("mask") is not None:
                    raw_mask = evidence["mask"]
                    mask_255 = (raw_mask * 255).astype(np.uint8) if raw_mask.max() <= 1 else raw_mask.astype(np.uint8)
                    st.markdown("#### 🔍 Change Detection Mask & Spatial Overlay:")
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.image(mask_255, caption="Binary Change Mask (White = Changed)", use_container_width=True)
                    with col_m2:
                        if "image_t2" in image_inputs:
                            try:
                                t2_pil = to_pil_rgb(image_inputs["image_t2"]).convert("RGBA")
                                mask_pil = Image.fromarray(mask_255).resize(t2_pil.size, Image.Resampling.NEAREST)
                                red_overlay = Image.new("RGBA", t2_pil.size, (255, 0, 0, 0))
                                red_patch = Image.new("RGBA", t2_pil.size, (255, 50, 50, 160))
                                red_overlay.paste(red_patch, mask=mask_pil)
                                blended = Image.alpha_composite(t2_pil, red_overlay)
                                st.image(blended, caption="T2 with Detected Changes (Red Overlay)", use_container_width=True)
                            except Exception:
                                st.image(mask_255, caption="Bi-Temporal Change Mask", use_container_width=True)
                        else:
                            st.image(mask_255, caption="Bi-Temporal Change Mask", use_container_width=True)
                
                # Provenance
                meta = result.get("metadata") or {}
                if meta:
                    with st.expander("ℹ️ Model Provenance & Parameters"):
                        st.json(meta)
                        
                # Execution Trace
                trace = result.get("trace") or {}
                if trace:
                    with st.expander("📋 Execution Trace & Audit Log"):
                        st.json(trace)
    else:
        st.write("Click **Run Analysis** in the sidebar to execute.")
