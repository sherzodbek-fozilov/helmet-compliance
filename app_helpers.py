# model list for the dropdown on the processing page
from __future__ import annotations

from pathlib import Path


def discover_models(project_root: Path) -> list[dict]:
    """models/*.pt first, then best.pt from local training runs"""
    found: list[Path] = sorted((project_root / "models").glob("*.pt"))
    found += sorted((project_root / "runs").glob("**/weights/best.pt"))
    models = []
    for p in found:
        label = p.name if p.parent.name == "models" else f"{p.parent.parent.name}/best.pt"
        models.append({"path": str(p), "label": label})
    return models
