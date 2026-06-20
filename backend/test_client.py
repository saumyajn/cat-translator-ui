"""Command-line test client for the cat intent API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


API_URL = "http://localhost:8000/translate"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="POST a local audio file to the cat intent API."
    )
    parser.add_argument("audio_file", help="Path to a local audio file, for example sample.wav")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)

    if not audio_path.exists():
        raise SystemExit(f"File not found: {audio_path}")

    if not audio_path.is_file():
        raise SystemExit(f"Path is not a file: {audio_path}")

    with audio_path.open("rb") as audio_file:
        response = requests.post(
            API_URL,
            files={"file": (audio_path.name, audio_file, "audio/wav")},
            timeout=60,
        )

    try:
        payload = response.json()
    except ValueError:
        response.raise_for_status()
        raise SystemExit(response.text)

    print(json.dumps(payload, indent=2))
    response.raise_for_status()


if __name__ == "__main__":
    main()
