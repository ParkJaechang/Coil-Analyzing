"""Workspace-local persistence for uploads, manifests, and reusable mappings."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from coil_analyzer.constants import (
    CONFIG_DIRNAME,
    EXPORT_DIRNAME,
    MANIFEST_FILENAME,
    MAPPING_LIBRARY_FILENAME,
    UPLOAD_DIRNAME,
    workspace_app_dir,
)
from coil_analyzer.models import DatasetMeta, RequestPoint
from coil_analyzer.utils.json_io import load_json, save_json


class WorkspaceStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.base_dir = workspace_app_dir(root)
        self.upload_dir = self.base_dir / UPLOAD_DIRNAME
        self.export_dir = self.base_dir / EXPORT_DIRNAME
        self.config_dir = self.base_dir / CONFIG_DIRNAME
        self.manifest_path = self.base_dir / MANIFEST_FILENAME
        self.mapping_library_path = self.base_dir / MAPPING_LIBRARY_FILENAME
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict[str, Any]:
        return load_json(
            self.manifest_path,
            {
                "datasets": [],
                "request_points": [],
                "analysis_window": {},
                "settings": {},
            },
        )

    def save_manifest(
        self,
        datasets: list[DatasetMeta],
        request_points: list[RequestPoint],
        analysis_window: dict[str, Any],
        settings: dict[str, Any],
    ) -> None:
        save_json(
            self.manifest_path,
            {
                "datasets": [item.to_dict() for item in datasets],
                "request_points": [item.to_dict() for item in request_points],
                "analysis_window": analysis_window,
                "settings": settings,
            },
        )

    def save_upload_bytes(self, file_name: str, raw_bytes: bytes) -> Path:
        suffix = Path(file_name).suffix.lower()
        target = self.upload_dir / f"{uuid.uuid4().hex}_{Path(file_name).stem}{suffix}"
        target.write_bytes(raw_bytes)
        return target

    def register_local_file(self, path: Path) -> Path:
        target = self.upload_dir / path.name
        if target.resolve() != path.resolve():
            shutil.copy2(path, target)
        return target

    def load_mapping_library(self) -> dict[str, Any]:
        return load_json(self.mapping_library_path, {})

    def save_mapping_library(self, payload: dict[str, Any]) -> None:
        save_json(self.mapping_library_path, payload)
