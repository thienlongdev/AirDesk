"""Kiểm tra cài đặt, không mở webcam và không gửi thao tác Windows."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def main() -> int:
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform != "win32" or sys.version_info[:2] != (3, 13) or struct.calcsize("P") != 8:
        print("LỖI: cần Windows và Python 3.13 x64.")
        return 1
    try:
        for name in ("cv2", "mediapipe", "PySide6.QtCore", "PySide6.QtWidgets", "numpy", "matplotlib", "PIL",
                     "airdesk.app", "airdesk.runtime", "airdesk.ui.keyboard_overlay"):
            importlib.import_module(name)
        from airdesk.resources import model_path, asset_path, resource_path
        for path in (asset_path("airdesk.ico"), asset_path("airdesk.png"),
                     resource_path("data", "words_vi.txt"), resource_path("data", "words_en.txt")):
            if not path.is_file():
                raise FileNotFoundError(f"Thiếu tài nguyên: {path}")
        if hashlib.sha256(model_path().read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError("Model thiếu hoặc sai SHA-256. Tải lại ZIP chính thức theo README.")
        from airdesk.config import CameraConfig
        from airdesk.vision.tracker import HandTracker
        tracker = HandTracker(CameraConfig())
        tracker.close()
    except Exception as exc:
        print(f"LỖI cài đặt: {exc}")
        return 1
    print("ĐẠT: thư viện, tài nguyên và model MediaPipe. Chưa kiểm tra webcam hoặc bàn tay thật.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
