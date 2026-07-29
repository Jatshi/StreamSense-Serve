from __future__ import annotations

import argparse
import json
from pathlib import Path

from streamsense.backends import BackendProfiles
from streamsense.model_registry import ModelManifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", type=Path, default=Path("configs/backends.json"))
    parser.add_argument("--models", type=Path, default=Path("models/serve_manifest.json"))
    args = parser.parse_args()
    profiles = BackendProfiles.load(args.backends)
    manifest = ModelManifest.model_validate_json(args.models.read_text(encoding="utf-8"))
    profile_names = {profile.name for profile in profiles.profiles}
    invalid = [
        model.model_id for model in manifest.models if model.backend_profile not in profile_names
    ]
    if invalid:
        raise ValueError(f"models reference missing backend profiles: {invalid}")
    print(
        json.dumps(
            {
                "status": "ok",
                "backend_profiles": len(profiles.profiles),
                "models": len(manifest.models),
                "default_profile": profiles.default_profile,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
