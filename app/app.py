"""
SatQuery AI — Streamlit GUI.
Owner: Person 4 (Agentic Orchestration + GUI).

Run:
    streamlit run app/app.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.utils.io_utils import load_config, save_uploaded_file
from src.agent.controller import AgentController

st.set_page_config(page_title="SatQuery AI", layout="wide")

config = load_config("configs/config.yaml")


@st.cache_resource
def get_controller():
    # TODO(Person 4): pass device="cuda" once models + GPU are available.
    return AgentController(config, device="cpu")


controller = get_controller()

st.title(config["app"]["title"])
st.caption("Agentic vision-language assistant for single, cross-modal, and "
           "bi-temporal remote-sensing imagery.")

st.sidebar.header("1. Choose input configuration")
input_mode = st.sidebar.radio(
    "Input type",
    ["Single image", "Optical + SAR pair", "Bi-temporal pair (two dates)"],
)

uploaded = {}
if input_mode == "Single image":
    uploaded["image"] = st.sidebar.file_uploader(
        "Upload image (GeoTIFF/TIFF/PNG/JPEG)",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
    )
elif input_mode == "Optical + SAR pair":
    uploaded["image_optical"] = st.sidebar.file_uploader(
        "Optical/multispectral image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="optical"
    )
    uploaded["image_sar"] = st.sidebar.file_uploader(
        "SAR image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="sar"
    )
else:
    uploaded["image_t1"] = st.sidebar.file_uploader(
        "Image at time T1", type=["tif", "tiff", "png", "jpg", "jpeg"], key="t1"
    )
    uploaded["image_t2"] = st.sidebar.file_uploader(
        "Image at time T2", type=["tif", "tiff", "png", "jpg", "jpeg"], key="t2"
    )

st.sidebar.header("2. Ask a question")
query = st.sidebar.text_area(
    "Query",
    placeholder="e.g. 'What changed between these two dates, and where did the change occur?'",
)

example_queries = [
    "Describe the land-cover and major objects visible in this image.",
    "Highlight the water body referred to in the query.",
    "What changed between these two dates, and where did the change occur?",
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
    "Has the built-up area increased, decreased, or remained unchanged?",
]
with st.sidebar.expander("Example queries"):
    for q in example_queries:
        st.write(f"- {q}")

run = st.sidebar.button("Run", type="primary")

col1, col2 = st.columns([1, 1])

if run:
    all_uploaded = [v for v in uploaded.values() if v is not None]
    if not all_uploaded or len(all_uploaded) != len(uploaded):
        st.error("Please upload all required image(s) for the selected input type.")
    elif not query.strip():
        st.error("Please enter a query.")
    else:
        # Save uploads to disk (models/controller work with file paths).
        saved_paths = {
            key: save_uploaded_file(file, target_dir="data/raw/uploads")
            for key, file in uploaded.items()
        }

        with col1:
            st.subheader("Input")
            for key, file in uploaded.items():
                st.image(file, caption=key, use_column_width=True)

        with st.spinner("Running agentic pipeline..."):
            result = controller.run(images=saved_paths, query=query)

        with col2:
            st.subheader("Result")
            if not result["success"]:
                st.error("Pipeline failed: " + "; ".join(result["errors"]))
            else:
                st.markdown(f"**Task selected:** `{result['task']}`")
                st.write(result["text"])
                st.progress(min(max(result["confidence"], 0.0), 1.0))
                st.caption(f"Confidence: {result['confidence']:.2f}  ·  Status: {result.get('status', '—')}")

                evidence = result.get("spatial_evidence") or {}
                if evidence.get("type") == "bbox" and evidence.get("bbox"):
                    st.write(f"Bounding box (in `{evidence.get('source', '?')}` frame): {evidence['bbox']}")
                if evidence.get("type") == "mask" and evidence.get("mask") is not None:
                    st.write(f"Mask generated in `{evidence.get('source', '?')}` frame "
                              "(see overlay — TODO: render).")

                metadata = result.get("metadata") or {}
                if metadata:
                    st.caption(
                        f"Model: {metadata.get('model', '—')} · "
                        f"Backbone: {metadata.get('backbone', '—')} · "
                        f"Checkpoint: {metadata.get('checkpoint', '—')}"
                    )

        st.subheader("Execution trace (auditable summary)")
        st.json(result["trace"])

        # TODO(Person 4): add a "Download report" button that renders
        # result + trace into a PDF via reportlab, saved under
        # app/static/reports/ and offered via st.download_button.
else:
    st.info("Upload image(s), enter a query, and click **Run** in the sidebar.")
