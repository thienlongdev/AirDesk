"""Windows 11-style Qt stylesheet (dark and light) and shared colours."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

PALETTES = {
    "dark": {
        "bg": "#1c1d22", "panel": "#25272e", "panel2": "#2c2f37", "border": "#383b45",
        "text": "#e8eaf0", "muted": "#9aa1ae", "accent": "#4c9aff", "accent_text": "#ffffff",
        "ok": "#3ccf8e", "warn": "#ffb64c", "bad": "#ff5c6c", "nav_sel": "#2f3542",
    },
    "light": {
        "bg": "#f3f4f7", "panel": "#ffffff", "panel2": "#f7f8fa", "border": "#dcdfe5",
        "text": "#1d2027", "muted": "#5f6672", "accent": "#0a64d8", "accent_text": "#ffffff",
        "ok": "#138a55", "warn": "#b46a00", "bad": "#c4283a", "nav_sel": "#e4ecfa",
    },
}

current: dict[str, str] = dict(PALETTES["dark"])


def qcolor(name: str) -> QColor:
    return QColor(current[name])


def apply(app: QApplication, theme: str) -> None:
    current.clear()
    current.update(PALETTES.get(theme, PALETTES["dark"]))
    c = current
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor(c["bg"]))
    pal.setColor(QPalette.WindowText, QColor(c["text"]))
    pal.setColor(QPalette.Base, QColor(c["panel2"]))
    pal.setColor(QPalette.Text, QColor(c["text"]))
    pal.setColor(QPalette.Button, QColor(c["panel2"]))
    pal.setColor(QPalette.ButtonText, QColor(c["text"]))
    pal.setColor(QPalette.Highlight, QColor(c["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor(c["accent_text"]))
    app.setPalette(pal)
    app.setStyleSheet(f"""
    * {{ font-family: "Segoe UI Variable Text", "Segoe UI"; font-size: 10pt; color: {c['text']}; }}
    QMainWindow, QDialog, #Page {{ background: {c['bg']}; }}
    QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
    #Nav {{ background: {c['panel']}; border: none; border-right: 1px solid {c['border']};
            padding: 8px 6px; outline: 0; }}
    #Nav::item {{ padding: 9px 12px; border-radius: 6px; margin: 1px 0; }}
    #Nav::item:selected {{ background: {c['nav_sel']}; color: {c['text']}; }}
    #Nav::item:hover:!selected {{ background: {c['panel2']}; }}
    #Brand {{ font-size: 15pt; font-weight: 600; padding: 4px 4px 0 4px; }}
    #Tagline {{ color: {c['muted']}; padding: 0 4px 8px 4px; }}
    #H1 {{ font-size: 18pt; font-weight: 600; }}
    #H2 {{ font-size: 11.5pt; font-weight: 600; }}
    #Muted {{ color: {c['muted']}; }}
    #Card {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 10px; }}
    #CardTitle {{ color: {c['muted']}; font-size: 8.5pt; }}
    #CardValue {{ font-size: 13pt; font-weight: 600; }}
    QGroupBox {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 10px;
                 margin-top: 14px; padding: 14px 12px 10px 12px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 12px; top: 2px; padding: 0 4px;
                        font-weight: 600; }}
    QPushButton {{ background: {c['panel2']}; border: 1px solid {c['border']}; border-radius: 6px;
                   padding: 6px 14px; }}
    QPushButton:hover {{ border-color: {c['accent']}; }}
    QPushButton:pressed {{ background: {c['border']}; }}
    QPushButton:disabled {{ color: {c['muted']}; }}
    QPushButton#Primary {{ background: {c['accent']}; color: {c['accent_text']}; border: none; font-weight: 600; }}
    QPushButton#Danger {{ background: {c['bad']}; color: #ffffff; border: none; font-weight: 600; }}
    QPushButton#Arm {{ font-size: 12pt; font-weight: 700; padding: 10px 22px; border-radius: 8px; border: none; }}
    QPushButton:checked {{ background: {c['accent']}; color: {c['accent_text']}; border: none; }}
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {c['panel2']}; border: 1px solid {c['border']};
                   border-radius: 6px; padding: 4px 8px; min-height: 20px; }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {c['accent']}; }}
    QComboBox QAbstractItemView {{ background: {c['panel']}; border: 1px solid {c['border']};
                   selection-background-color: {c['accent']}; }}
    QSlider::groove:horizontal {{ height: 4px; background: {c['border']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {c['accent']}; width: 14px; height: 14px; margin: -6px 0;
                   border-radius: 7px; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; }}
    QTableWidget, QListWidget, QPlainTextEdit, QTextEdit {{ background: {c['panel']}; border: 1px solid {c['border']};
                   border-radius: 8px; gridline-color: {c['border']}; }}
    QHeaderView::section {{ background: {c['panel2']}; border: none; border-bottom: 1px solid {c['border']};
                   padding: 4px 8px; font-weight: 600; }}
    QToolTip {{ background: {c['panel']}; border: 1px solid {c['border']}; padding: 4px; }}
    QProgressBar {{ background: {c['panel2']}; border: none; border-radius: 4px; height: 8px; text-align: center; }}
    QProgressBar::chunk {{ background: {c['accent']}; border-radius: 4px; }}
    """)
