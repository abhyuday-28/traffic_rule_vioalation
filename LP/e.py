from __future__ import annotations

import base64
import os
from pathlib import Path

import requests


OPENALPR_SECRET_KEY = os.getenv("OPENALPR_SECRET_KEY", "")


def recognize_plate(image_path: str | os.PathLike[str]) -> str | None:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not OPENALPR_SECRET_KEY:
        raise RuntimeError("Set the OPENALPR_SECRET_KEY environment variable before calling this helper.")

    with image_path.open("rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())

    url = (
        "https://api.openalpr.com/v2/recognize_bytes"
        f"?recognize_vehicle=1&country=ind&secret_key={OPENALPR_SECRET_KEY}"
    )
    response = requests.post(url, data=encoded_image, timeout=30)
    response.raise_for_status()

    results = response.json().get("results", [])
    if not results:
        return None
    return results[0].get("plate")


if __name__ == "__main__":
    sample_image = Path(__file__).resolve().parent / "frame0.jpg"
    try:
        plate = recognize_plate(sample_image)
        print(plate or "No number plate found")
    except Exception as exc:
        print(exc)
