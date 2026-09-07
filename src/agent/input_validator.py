"""
Input validator — checks number, modality, format, metadata, and
co-registration compatibility of uploaded images before task routing.
Owner: Person 4 (Agentic Orchestration + GUI).
"""

from src.preprocessing.geotiff_utils import read_image, check_coregistration, extract_metadata

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def validate_input(images: dict, config: dict) -> dict:
    """images: one of
        {"image": path}                                -> single image
        {"image_optical": path, "image_sar": path}      -> cross-modal pair
        {"image_t1": path, "image_t2": path}             -> bi-temporal pair

    Returns:
        {
          "valid": bool,
          "mode": "single" | "cross_modal" | "bi_temporal" | None,
          "errors": list[str],
          "metadata": dict,   # per-image metadata, keyed same as `images`
        }
    """
    errors = []
    metadata = {}

    keys = set(images.keys())
    if keys == {"image"}:
        mode = "single"
    elif keys == {"image_optical", "image_sar"}:
        mode = "cross_modal"
    elif keys == {"image_t1", "image_t2"}:
        mode = "bi_temporal"
    else:
        return {"valid": False, "mode": None,
                "errors": [f"Unrecognized image key combination: {keys}"],
                "metadata": {}}

    # Format check
    for key, path in images.items():
        ext = "." + path.lower().rsplit(".", 1)[-1]
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"{key}: unsupported format '{ext}'")

    if errors:
        return {"valid": False, "mode": mode, "errors": errors, "metadata": {}}

    # TODO(Person 4): load each image via read_image(), populate metadata,
    # and for cross_modal/bi_temporal modes call check_coregistration() and
    # append an error if images are not co-registered.
    #
    # for key, path in images.items():
    #     img = read_image(path)
    #     metadata[key] = extract_metadata(img)
    #
    # if mode in ("cross_modal", "bi_temporal"):
    #     a, b = list(images.values())
    #     result = check_coregistration(read_image(a), read_image(b))
    #     if not result["co_registered"]:
    #         errors.append(result["reason"])

    return {"valid": len(errors) == 0, "mode": mode, "errors": errors, "metadata": metadata}
