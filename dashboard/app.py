import html
import textwrap

import numpy as np
import streamlit as st
from PIL import Image

from src.post_weld.yolo_detector import run_post_weld_model
from src.pre_weld.crack_predictor import predict_crack_risk
from src.pre_weld.process_config import (
    get_process_config,
    get_process_names,
)
from src.pre_weld.recommendation_engine import (
    recommend_parameter_changes,
)
from src.pre_weld.weldright_adapter import (
    predict_weldright,
)
from src.risk_engine import calculate_severity, preliminary_risk


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Material Integrity Assessment in Welding",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# HTML HELPER
# ============================================================

def render_html(content):
    st.html(textwrap.dedent(content).strip())


# ============================================================
# GLOBAL STYLING
# ============================================================

render_html(
    """
    <style>

    /* ========================================================
       GLOBAL THEME
       ======================================================== */

    :root {
        color-scheme: dark;
        --mi-bg: #070b12;
        --mi-bg-soft: #0b111a;
        --mi-card: #101925;
        --mi-card-2: #0d1621;
        --mi-border: #243246;
        --mi-border-soft: #1b293b;
        --mi-text: #f8fafc;
        --mi-text-soft: #cbd5e1;
        --mi-muted: #8fa0b5;
        --mi-dim: #64748b;
        --mi-blue: #3b82f6;
        --mi-cyan: #06b6d4;
        --mi-green: #22c55e;
        --mi-yellow: #f59e0b;
        --mi-red: #ef4444;
    }


    /* ========================================================
       STREAMLIT CHROME
       ======================================================== */

    [data-testid="stHeader"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ========================================================
       GLOBAL APP BACKGROUND
       ======================================================== */

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"],
    .stApp,
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background-color: var(--mi-bg) !important;
        color: var(--mi-text) !important;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 4% 0%,
                rgba(37, 99, 235, 0.14),
                transparent 28%
            ),
            radial-gradient(
                circle at 96% 2%,
                rgba(6, 182, 212, 0.08),
                transparent 28%
            ),
            linear-gradient(
                180deg,
                #070b12 0%,
                #080e16 48%,
                #070b12 100%
            ) !important;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2.4rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }


    /* ========================================================
       STREAMLIT ELEMENT BACKGROUNDS
       ======================================================== */

    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="column"] {
        color: var(--mi-text);
    }

    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"] {
        color: var(--mi-text-soft);
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--mi-muted) !important;
    }


    /* ========================================================
       TYPOGRAPHY
       ======================================================== */

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: var(--mi-text) !important;
        font-weight: 850 !important;
    }

    p,
    li,
    span,
    label {
        color: inherit;
    }

    .stMarkdown p {
        color: var(--mi-text-soft);
    }

    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4 {
        color: var(--mi-text) !important;
    }


    /* ========================================================
       BRAND
       ======================================================== */

    .brand-title {
        color: #ffffff;
        font-size: clamp(38px, 4vw, 56px);
        font-weight: 950;
        letter-spacing: -2.2px;
        line-height: 1.02;
    }

    .brand-subtitle {
        color: #8fa0b5;
        margin-top: 12px;
        font-size: 15px;
        font-weight: 500;
        line-height: 1.6;
    }

    .section-kicker {
        color: #60a5fa;
        font-size: 11px;
        font-weight: 950;
        letter-spacing: 2px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }

    .page-title {
        color: #ffffff;
        font-size: clamp(32px, 3.5vw, 48px);
        font-weight: 950;
        letter-spacing: -1.5px;
        line-height: 1.05;
        margin: 0 0 12px 0;
    }

    .page-description {
        color: #94a3b8;
        font-size: 16px;
        line-height: 1.75;
        max-width: 920px;
    }


    /* ========================================================
       ONLINE STATUS
       ======================================================== */

    .online-pill {
        display: inline-block;
        padding: 9px 15px;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.32);
        color: #86efac;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 0.9px;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-section {
        padding: 48px 20px 36px 20px;
        text-align: center;
    }

    .hero-kicker {
        color: #60a5fa;
        font-size: 12px;
        font-weight: 950;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    .hero-title {
        color: #ffffff;
        font-size: clamp(42px, 6vw, 72px);
        font-weight: 950;
        letter-spacing: -3px;
        line-height: 0.98;
        margin-bottom: 24px;
    }

    .hero-accent {
        color: #60a5fa;
    }

    .hero-text {
        max-width: 980px;
        margin: 0 auto;
        color: #aab7c8;
        font-size: 17px;
        line-height: 1.9;
    }


    /* ========================================================
       INTRO PANEL
       ======================================================== */

    .intro-panel {
        background:
            linear-gradient(
                145deg,
                rgba(17, 29, 45, 0.96),
                rgba(10, 18, 29, 0.96)
            );
        border: 1px solid #26374d;
        border-radius: 20px;
        padding: 28px 30px;
        margin-top: 20px;
        margin-bottom: 34px;
        box-shadow:
            0 14px 40px rgba(0, 0, 0, 0.16);
    }

    .intro-heading {
        color: #ffffff;
        font-size: 21px;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .intro-text {
        color: #aebbd0;
        font-size: 15px;
        line-height: 1.8;
    }


    /* ========================================================
       MODE CARDS
       ======================================================== */

    .mode-card {
        min-height: 285px;
        padding: 27px;
        border-radius: 20px;
        border: 1px solid #29394e;
        background:
            linear-gradient(
                145deg,
                #111c2a,
                #0c141e
            );
        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.14);
        transition:
            transform 0.18s ease,
            border-color 0.18s ease,
            box-shadow 0.18s ease;
    }

    .mode-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
        box-shadow:
            0 18px 42px rgba(0, 0, 0, 0.22);
    }

    .mode-number {
        color: #5d8fcf;
        font-size: 11px;
        font-weight: 950;
        letter-spacing: 2px;
        margin-bottom: 22px;
    }

    .mode-icon {
        font-size: 34px;
        margin-bottom: 16px;
    }

    .mode-title {
        color: #ffffff;
        font-size: 24px;
        font-weight: 950;
        letter-spacing: -0.5px;
        margin-bottom: 12px;
    }

    .mode-description {
        color: #9cabc0;
        font-size: 14px;
        line-height: 1.7;
        min-height: 72px;
    }

    .mode-status {
        display: inline-block;
        margin-top: 18px;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 1px;
    }

    .status-ready {
        color: #86efac;
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.25);
    }

    .status-integration {
        color: #fcd34d;
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
    }


    /* ========================================================
       WORKFLOW
       ======================================================== */

    .workflow-panel {
        margin-top: 42px;
        padding: 28px;
        border-radius: 20px;
        border: 1px solid #25354a;
        background:
            linear-gradient(
                145deg,
                rgba(12, 20, 31, 0.92),
                rgba(9, 16, 25, 0.92)
            );
    }

    .workflow-title {
        color: #ffffff;
        font-size: 20px;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .workflow-step {
        padding: 17px;
        border-radius: 13px;
        background: #111b27;
        border: 1px solid #26374b;
        text-align: center;
        transition:
            border-color 0.18s ease,
            transform 0.18s ease;
    }

    .workflow-step:hover {
        border-color: #315a87;
        transform: translateY(-1px);
    }

    .workflow-step-number {
        color: #60a5fa;
        font-size: 11px;
        font-weight: 950;
        letter-spacing: 1px;
    }

    .workflow-step-title {
        color: #f8fafc;
        font-size: 14px;
        font-weight: 850;
        margin-top: 7px;
    }


    /* ========================================================
       NAVIGATION
       ======================================================== */

    .nav-label {
        color: #8191a6;
        font-size: 10px;
        font-weight: 950;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .nav-current {
        color: #ffffff;
        font-size: 18px;
        font-weight: 900;
    }


    /* ========================================================
       RESULT HEADINGS
       ======================================================== */

    .result-heading {
        color: #ffffff;
        font-size: clamp(30px, 3vw, 42px);
        font-weight: 950;
        line-height: 1.1;
        margin: 4px 0 22px 0;
    }


    /* ========================================================
       SUMMARY CARDS
       ======================================================== */

    .summary-card {
        background:
            linear-gradient(
                145deg,
                #121b28,
                #0d141e
            );
        border: 1px solid #293648;
        border-radius: 16px;
        padding: 22px;
        min-height: 145px;
        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.12);
    }

    .summary-label {
        color: #91a0b4;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 1.2px;
        margin-bottom: 13px;
    }

    .summary-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 950;
        line-height: 1.12;
        overflow-wrap: anywhere;
    }

    .summary-value.model-name {
        font-size: 23px;
        line-height: 1.15;
        overflow-wrap: normal;
        word-break: normal;
        white-space: normal;
        letter-spacing: -0.4px;
    }

    .summary-sub {
        color: #8290a3;
        margin-top: 9px;
        font-size: 13px;
        line-height: 1.4;
    }


    /* ========================================================
       RESULT COLORS
       ======================================================== */

    .low {
        color: #62e6a4 !important;
    }

    .medium {
        color: #ffd166 !important;
    }

    .high {
        color: #ff9869 !important;
    }

    .critical {
        color: #ff5d68 !important;
    }

    .uncertain {
        color: #c5a6ff !important;
    }


    /* ========================================================
       CUSTOM BANNERS
       ======================================================== */

    .banner {
        border-radius: 14px;
        padding: 18px 20px;
        font-size: 15px;
        line-height: 1.65;
    }

    .banner-good {
        background:
            linear-gradient(
                135deg,
                rgba(34, 197, 94, 0.10),
                rgba(16, 185, 129, 0.045)
            );
        border: 1px solid rgba(34, 197, 94, 0.30);
        color: #bff3d3;
    }

    .banner-warning {
        background:
            linear-gradient(
                135deg,
                rgba(245, 158, 11, 0.10),
                rgba(234, 179, 8, 0.04)
            );
        border: 1px solid rgba(245, 158, 11, 0.32);
        color: #ffe0a4;
    }

    .banner-danger {
        background:
            linear-gradient(
                135deg,
                rgba(239, 68, 68, 0.11),
                rgba(220, 38, 38, 0.045)
            );
        border: 1px solid rgba(239, 68, 68, 0.32);
        color: #ffc2c7;
    }


    /* ========================================================
       PANELS
       ======================================================== */

    .panel {
        background:
            linear-gradient(
                145deg,
                #101925,
                #0d151f
            );
        border: 1px solid #293648;
        border-radius: 16px;
        padding: 21px 23px;
        color: #d9e2ee;
        font-size: 16px;
        line-height: 1.75;
        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.10);
    }

    .context-label {
        color: #8191a6;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 1.2px;
        margin-bottom: 5px;
    }

    .context-value {
        color: #f1f5f9;
        font-size: 16px;
        font-weight: 650;
    }


    /* ========================================================
       DEFECT ROWS
       ======================================================== */

    .detection-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 25px;
        padding: 12px 0;
        border-bottom: 1px solid #24303e;
    }

    .detection-row:last-child {
        border-bottom: none;
    }

    .detection-name {
        color: #f1f5f9;
        font-weight: 750;
    }

    .detection-confidence {
        color: #a7b6c9;
        font-weight: 850;
    }


    /* ========================================================
       RECOMMENDATION CARDS
       ======================================================== */

    .recommendation-card {
        background:
            linear-gradient(
                145deg,
                #111c2a,
                #0d151f
            );
        border: 1px solid #2b3b50;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 14px;
        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.11);
    }

    .recommendation-title {
        color: #ffffff;
        font-size: 17px;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .recommendation-metric {
        color: #9fb0c4;
        font-size: 13px;
        margin-bottom: 7px;
    }

    .change-row {
        padding: 9px 0;
        border-bottom: 1px solid #223043;
    }

    .change-row:last-child {
        border-bottom: none;
    }

    .change-name {
        color: #dbe7f5;
        font-weight: 750;
    }

    .change-values {
        color: #9fb0c4;
    }


    /* ========================================================
       INPUT LABELS
       ======================================================== */

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #dbe5f1 !important;
        font-size: 14px !important;
        font-weight: 750 !important;
    }

    [data-testid="stCaptionContainer"] {
        color: #8494a8 !important;
    }


    /* ========================================================
       SELECTBOX - CLOSED STATE
       ======================================================== */

    div[data-baseweb="select"] {
        background: transparent !important;
    }

    div[data-baseweb="select"] > div {
        background: #111b27 !important;
        border: 1px solid #334155 !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
        min-height: 44px !important;
        box-shadow: none !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #4b6380 !important;
    }

    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }

    div[data-baseweb="select"] input {
        color: #ffffff !important;
        caret-color: #60a5fa !important;
    }

    div[data-baseweb="select"] svg {
        fill: #94a3b8 !important;
        color: #94a3b8 !important;
    }


    /* ========================================================
       SELECTBOX - DROPDOWN POPUP
       THIS FIXES THE WHITE BACKGROUND / WHITE TEXT ISSUE
       ======================================================== */

    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"] {
        background: #101925 !important;
        color: #f8fafc !important;
        border-color: #334155 !important;
    }

    div[data-baseweb="popover"] {
        border-radius: 12px !important;
        box-shadow:
            0 18px 45px rgba(0, 0, 0, 0.45) !important;
    }

    div[role="option"],
    li[role="option"] {
        background: #101925 !important;
        color: #e5edf7 !important;
        min-height: 40px !important;
    }

    div[role="option"] *,
    li[role="option"] * {
        color: #e5edf7 !important;
    }

    div[role="option"]:hover,
    li[role="option"]:hover {
        background: #18273a !important;
        color: #ffffff !important;
    }

    div[role="option"][aria-selected="true"],
    li[role="option"][aria-selected="true"] {
        background: #17283c !important;
        color: #60a5fa !important;
    }

    div[role="option"][aria-selected="true"] *,
    li[role="option"][aria-selected="true"] * {
        color: #60a5fa !important;
    }


    /* ========================================================
       NUMBER INPUT
       ======================================================== */

    [data-testid="stNumberInput"] {
        background: transparent !important;
    }

    [data-testid="stNumberInput"] > div {
        background: transparent !important;
    }

    [data-testid="stNumberInput"] input {
        background: #111b27 !important;
        color: #ffffff !important;
        font-size: 16px !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }

    [data-testid="stNumberInput"] input:focus {
        border-color: #3b82f6 !important;
        box-shadow:
            0 0 0 1px #3b82f6 !important;
    }

    [data-testid="stNumberInput"] button {
        background: #111b27 !important;
        color: #cbd5e1 !important;
        border-color: #334155 !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background: #18273a !important;
        color: #ffffff !important;
    }

    [data-testid="stNumberInput"] button svg {
        fill: #cbd5e1 !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        color: #dbe5f1 !important;
    }

    [data-testid="stFileUploader"] section {
        background:
            linear-gradient(
                145deg,
                #111b27,
                #0d151f
            ) !important;
        border: 1px dashed #40526a !important;
        border-radius: 12px !important;
        padding: 18px !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #4f7fb5 !important;
    }

    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] label {
        color: #cbd5e1 !important;
    }

    [data-testid="stFileUploader"] button {
        background: #1a2738 !important;
        color: #f8fafc !important;
        border: 1px solid #3a4c63 !important;
        border-radius: 9px !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background: #22344a !important;
        border-color: #4c6b8e !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        min-height: 45px;
        border-radius: 11px !important;
        font-weight: 850 !important;
        background: #121d2b !important;
        color: #e8eef7 !important;
        border: 1px solid #334155 !important;
        box-shadow: none !important;
        transition:
            background 0.18s ease,
            border-color 0.18s ease,
            transform 0.18s ease;
    }

    .stButton > button:hover {
        background: #1a2a3e !important;
        color: #ffffff !important;
        border-color: #4b6380 !important;
        transform: translateY(-1px);
    }

    .stButton > button:focus {
        outline: none !important;
        box-shadow:
            0 0 0 2px rgba(59, 130, 246, 0.28) !important;
    }

    .stButton > button[kind="primary"] {
        width: 100%;
        min-height: 55px;
        border: none !important;
        border-radius: 12px !important;
        background:
            linear-gradient(
                90deg,
                #2563eb,
                #0891b2
            ) !important;
        color: #ffffff !important;
        font-size: 15px !important;
        font-weight: 900 !important;
        letter-spacing: 0.4px;
        box-shadow:
            0 8px 22px rgba(37, 99, 235, 0.18) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background:
            linear-gradient(
                90deg,
                #3b82f6,
                #06b6d4
            ) !important;
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow:
            0 10px 28px rgba(37, 99, 235, 0.25) !important;
    }

    .stButton > button:disabled {
        background: #111923 !important;
        color: #64748b !important;
        border-color: #263445 !important;
        opacity: 0.72 !important;
        transform: none !important;
    }


    /* ========================================================
       SLIDER
       ======================================================== */

    [data-testid="stSlider"] {
        color: #dbe5f1 !important;
    }

    [data-testid="stSlider"] label {
        color: #dbe5f1 !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: #60a5fa !important;
        border-color: #60a5fa !important;
    }


    /* ========================================================
       ALERTS / NATIVE STREAMLIT MESSAGES
       THIS PREVENTS LIGHT ALERT BOXES
       ======================================================== */

    [data-testid="stAlert"] {
        background: #101925 !important;
        color: #dbe5f1 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }


    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div {
        color: #dbe5f1 !important;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    [data-testid="stExpander"] {
        background: #0f1823 !important;
        border: 1px solid #293648 !important;
        border-radius: 13px !important;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        background: #0f1823 !important;
        color: #e5edf7 !important;
    }

    [data-testid="stExpander"] summary:hover {
        background: #131f2e !important;
    }

    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        color: #e5edf7 !important;
    }

    [data-testid="stExpanderDetails"] {
        background: #0c141e !important;
        color: #cbd5e1 !important;
    }

    [data-testid="stExpanderDetails"] p,
    [data-testid="stExpanderDetails"] li {
        color: #cbd5e1 !important;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: #243244 !important;
        opacity: 0.9 !important;
    }


    /* ========================================================
       IMAGES
       ======================================================== */

    [data-testid="stImage"] {
        background: #0d151f !important;
        border-radius: 13px !important;
    }

    [data-testid="stImage"] img {
        max-height: 520px;
        object-fit: contain;
        border-radius: 12px;
    }


    /* ========================================================
       SPINNER
       ======================================================== */

    [data-testid="stSpinner"] {
        color: #93c5fd !important;
    }


    /* ========================================================
       TOOLTIP / POPOVER TEXT
       ======================================================== */

    [data-baseweb="tooltip"] {
        background: #101925 !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
    }

    [data-baseweb="tooltip"] * {
        color: #f8fafc !important;
    }


    /* ========================================================
       SCROLLBAR
       ======================================================== */

    ::-webkit-scrollbar {
        width: 9px;
        height: 9px;
    }

    ::-webkit-scrollbar-track {
        background: #080d14;
    }

    ::-webkit-scrollbar-thumb {
        background: #26364a;
        border-radius: 999px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #38506b;
    }


    /* ========================================================
       RESPONSIVE
       ======================================================== */

    @media (max-width: 800px) {

        .block-container {
            padding-top: 1.8rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .hero-section {
            padding-left: 5px;
            padding-right: 5px;
        }

        .hero-title {
            font-size: 42px !important;
            letter-spacing: -2px;
        }

        .brand-title {
            font-size: 34px !important;
        }

        .summary-value {
            font-size: 23px !important;
        }

        .summary-value.model-name {
            font-size: 20px !important;
        }

        .mode-card {
            min-height: auto;
        }

        .workflow-panel {
            padding: 20px;
        }
    }

    </style>
    """
)


# ============================================================
# HELPERS
# ============================================================

def level_css(level):
    return {
        "LOW": "low",
        "MEDIUM": "medium",
        "HIGH": "high",
        "CRITICAL": "critical",
        "UNCERTAIN": "uncertain",
    }.get(level, "")


def confidence_status(confidence):
    pct = confidence * 100

    if pct >= 70:
        return "HIGH", "Strong AI indication"

    if pct >= 50:
        return "MODERATE", "Moderate AI indication"

    if pct >= 30:
        return "LOW", "Weak AI indication"

    return "VERY LOW", "Unverified AI indication"


def display_model_name(model_type):
    model_names = {
        "RandomForestClassifier": "Random Forest",
        "RandomForest": "Random Forest",
    }

    return model_names.get(
        model_type,
        str(model_type),
    )


def summary_card(
    label,
    value,
    subtitle,
    level=None,
    model_name=False,
):
    css = level_css(level)

    if model_name:
        css = f"{css} model-name".strip()

    render_html(
        f"""
        <div class="summary-card">

            <div class="summary-label">
                {html.escape(str(label))}
            </div>

            <div class="summary-value {css}">
                {html.escape(str(value))}
            </div>

            <div class="summary-sub">
                {html.escape(str(subtitle))}
            </div>

        </div>
        """
    )


# ============================================================
# NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


def go_home():
    st.session_state.page = "home"
    st.rerun()


def go_to(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# ENGINEERING DATABASE
# ============================================================

DEFECT_DB = {

    "Burn Through-BT": {
        "severity": "HIGH",
        "causes": [
            "Excessive heat input",
            "Welding current may be too high",
            "Travel speed may be too low",
            "Excessive root gap or unsuitable joint preparation",
        ],
        "verification": [
            "Perform visual and dimensional inspection",
            "Assess penetration and remaining thickness",
            "Use suitable NDT where required",
        ],
        "recommendations": [
            "Review welding current and heat input",
            "Inspect the complete affected region",
            "Repair or re-weld if acceptance limits are exceeded",
        ],
    },

    "Contamination-CN": {
        "severity": "MEDIUM",
        "causes": [
            "Oil, oxide, rust or moisture contamination",
            "Poor surface preparation",
            "Contaminated filler material",
            "Inadequate shielding",
        ],
        "verification": [
            "Clean and visually inspect the region",
            "Review shielding conditions",
            "Inspect nearby weld surface",
        ],
        "recommendations": [
            "Improve surface preparation",
            "Check filler and shielding conditions",
            "Reinspect after corrective action",
        ],
    },

    "Lack of Fusion-LOF": {
        "severity": "HIGH",
        "causes": [
            "Insufficient heat input",
            "Excessive travel speed",
            "Incorrect torch or electrode angle",
            "Poor joint preparation",
        ],
        "verification": [
            "Use UT or RT inspection where applicable",
            "Determine defect location and extent",
            "Review joint preparation",
        ],
        "recommendations": [
            "Review welding procedure parameters",
            "Repair the affected region where required",
            "Reinspect after repair",
        ],
    },

    "Normal Weld-NW": {
        "severity": "LOW",
        "causes": [
            "No major visible surface defect indicated by the current model",
        ],
        "verification": [
            "Perform routine visual inspection",
            "Apply normal NDT requirements when applicable",
        ],
        "recommendations": [
            "Continue standard inspection procedures",
            "Record the inspection result",
        ],
    },

    "Spatter-SR": {
        "severity": "LOW",
        "causes": [
            "Excessive welding current",
            "Incorrect voltage",
            "Unstable welding arc",
            "Shielding instability",
        ],
        "verification": [
            "Inspect after cleaning the surface",
            "Check whether spatter obscures other discontinuities",
        ],
        "recommendations": [
            "Remove excessive spatter",
            "Review current and voltage",
            "Check welding arc stability",
        ],
    },

    "Surface Crack-SC": {
        "severity": "CRITICAL",
        "causes": [
            "Residual welding stresses",
            "Rapid cooling",
            "Hydrogen-assisted cracking",
            "Fatigue or stress concentration",
        ],
        "verification": [
            "Perform suitable NDT",
            "Determine crack depth and extent",
            "Obtain qualified engineering evaluation",
        ],
        "recommendations": [
            "Do not rely only on visual inspection",
            "Assess before continued service",
            "Repair or replace affected region where required",
        ],
    },

    "Undercut-UC": {
        "severity": "MEDIUM",
        "causes": [
            "Excessive welding current",
            "High travel speed",
            "Incorrect torch or electrode angle",
            "Improper welding technique",
        ],
        "verification": [
            "Measure actual undercut depth",
            "Compare with applicable acceptance criteria",
            "Inspect the entire weld length",
        ],
        "recommendations": [
            "Adjust welding current and travel speed",
            "Correct torch or electrode angle",
            "Repair if allowable limits are exceeded",
        ],
    },
}


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([5, 1])

with header_left:

    render_html(
        """
        <div class="brand-title">
            Material Integrity AI
        </div>

        <div class="brand-subtitle">
            AI-Assisted Weld Inspection • Root-Cause Support •
            Engineering Decision Support
        </div>
        """
    )

with header_right:

    render_html(
        """
        <div style="text-align:right; padding-top:8px;">

            <span class="online-pill">
                ● MODEL PLATFORM
            </span>

        </div>
        """
    )


st.divider()


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    render_html(
        """
        <div class="hero-section">

            <div class="hero-kicker">
                WELDING • MATERIAL INTEGRITY • ARTIFICIAL INTELLIGENCE
            </div>

            <div class="hero-title">
                Material Integrity<br>
                <span class="hero-accent">
                    Assessment in Welding
                </span>
            </div>

            <div class="hero-text">
                Material Integrity Assessment in Welding is an
                AI-assisted approach for identifying and assessing
                welding-related defect risk across the complete
                welding lifecycle — before welding, during the
                welding process, and after welding.
            </div>

        </div>
        """
    )

    render_html(
        """
        <div class="intro-panel">

            <div class="intro-heading">
                What does Material Integrity Assessment mean?
            </div>

            <div class="intro-text">
                Welding quality is influenced by process parameters,
                material conditions, welding behaviour and the final
                condition of the welded component. A defect can
                therefore be considered at different stages rather
                than only after the weld has been completed.
                <br><br>
                This platform is designed as an AI-assisted
                engineering decision-support system. It brings
                together predictive analysis before welding,
                in-process defect monitoring during welding, and
                visual defect assessment after welding.
                <br><br>
                The objective is to help engineers identify potential
                problems earlier, understand possible defect-related
                factors, and support better inspection and parameter
                decisions. AI outputs are treated as decision-support
                indications and should be verified using applicable
                welding procedures, inspection standards and qualified
                engineering judgement.
            </div>

        </div>
        """
    )

    render_html(
        """
        <div class="section-kicker">
            WELDING INTEGRITY WORKFLOW
        </div>

        <div class="page-title">
            Three stages. One integrity assessment.
        </div>

        <div class="page-description">
            Select the stage you want to analyse.
        </div>
        """
    )

    st.write("")

    c1, c2, c3 = st.columns(
        3,
        gap="large",
    )

    with c1:

        render_html(
            """
            <div class="mode-card">

                <div class="mode-number">
                    01 / BEFORE WELDING
                </div>

                <div class="mode-icon">
                    ⚡
                </div>

                <div class="mode-title">
                    Pre-Weld Prediction
                </div>

                <div class="mode-description">
                    Enter welding-process parameters and estimate
                    defect risk before the weld is made. The current
                    implemented model evaluates crack risk for the
                    available Laser Welding / WSteel experiment.
                </div>

                <span class="mode-status status-ready">
                    MODEL AVAILABLE
                </span>

            </div>
            """
        )

        st.write("")

        if st.button(
            "OPEN PRE-WELD PREDICTION →",
            key="home_pre",
            type="primary",
            use_container_width=True,
        ):
            go_to("pre")

    with c2:

        render_html(
            """
            <div class="mode-card">

                <div class="mode-number">
                    02 / DURING WELDING
                </div>

                <div class="mode-icon">
                    📡
                </div>

                <div class="mode-title">
                    In-Process Prediction
                </div>

                <div class="mode-description">
                    Monitor welding behaviour and identify potential
                    defects while the welding operation is taking
                    place. This section is reserved for the separate
                    in-process model being developed for the project.
                </div>

                <span class="mode-status status-integration">
                    MODEL INTEGRATION PENDING
                </span>

            </div>
            """
        )

        st.write("")

        if st.button(
            "OPEN IN-PROCESS MODULE →",
            key="home_inprocess",
            use_container_width=True,
        ):
            go_to("inprocess")

    with c3:

        render_html(
            """
            <div class="mode-card">

                <div class="mode-number">
                    03 / AFTER WELDING
                </div>

                <div class="mode-icon">
                    🔍
                </div>

                <div class="mode-title">
                    Post-Weld Detection
                </div>

                <div class="mode-description">
                    Upload a completed-weld image and use the trained
                    YOLO-based detector to identify visible weld
                    defect indications and obtain preliminary
                    engineering guidance.
                </div>

                <span class="mode-status status-ready">
                    MODEL AVAILABLE
                </span>

            </div>
            """
        )

        st.write("")

        if st.button(
            "OPEN POST-WELD INSPECTION →",
            key="home_post",
            type="primary",
            use_container_width=True,
        ):
            go_to("post")

    render_html(
        """
        <div class="workflow-panel">

            <div class="workflow-title">
                From prevention to inspection
            </div>

            <div style="
                color:#9cabc0;
                line-height:1.7;
                margin-bottom:20px;
            ">
                The intended platform workflow is to identify potential
                risk as early as possible, monitor the welding operation,
                and finally verify the completed weld.
            </div>

        </div>
        """
    )

    w1, w2, w3, w4 = st.columns(4)

    with w1:
        render_html(
            """
            <div class="workflow-step">
                <div class="workflow-step-number">
                    STEP 01
                </div>
                <div class="workflow-step-title">
                    Enter Parameters
                </div>
            </div>
            """
        )

    with w2:
        render_html(
            """
            <div class="workflow-step">
                <div class="workflow-step-number">
                    STEP 02
                </div>
                <div class="workflow-step-title">
                    Predict Risk
                </div>
            </div>
            """
        )

    with w3:
        render_html(
            """
            <div class="workflow-step">
                <div class="workflow-step-number">
                    STEP 03
                </div>
                <div class="workflow-step-title">
                    Monitor Process
                </div>
            </div>
            """
        )

    with w4:
        render_html(
            """
            <div class="workflow-step">
                <div class="workflow-step-number">
                    STEP 04
                </div>
                <div class="workflow-step-title">
                    Inspect Final Weld
                </div>
            </div>
            """
        )

    st.write("")
    st.write("")

    render_html(
        """
        <div style="
            text-align:center;
            color:#64748b;
            font-size:13px;
            line-height:1.7;
            max-width:900px;
            margin:0 auto;
        ">
            Material Integrity AI is an AI-assisted engineering
            prototype. Model predictions and recommendations do not
            replace WPS requirements, qualified inspection,
            non-destructive testing or engineering judgement.
        </div>
        """
    )


# ============================================================
# PRE-WELD PAGE
# ============================================================

elif st.session_state.page == "pre":

    if st.button(
        "← Back to Home",
        key="pre_home",
    ):
        go_home()

    st.write("")

    render_html(
        """
        <div class="section-kicker">
            STAGE 01 / BEFORE WELDING
        </div>

        <div class="page-title">
            Pre-Weld Defect Prediction
        </div>

        <div class="page-description">
            Select a welding process, enter process-specific
            parameters, and use the available trained model to
            estimate defect risk before welding.
        </div>
        """
    )

    st.write("")

    process_names = get_process_names()

    process_options = [
        "WeldRight Industrial Screening",
        *process_names,
    ]

    selected_process = st.selectbox(
        "Welding Process",
        process_options,
        key="pre_process",
    )


    # ========================================================
    # WELDRIGHT INDUSTRIAL SCREENING
    # ========================================================

    if selected_process == "WeldRight Industrial Screening":

        render_html(
            """
            <div class="panel">

                <div class="context-label">
                    SELECTED MODE
                </div>

                <div class="context-value">
                    WeldRight Industrial Screening
                </div>

                <br>

                <div class="context-label">
                    PREDICTION TYPE
                </div>

                <div class="context-value">
                    Defect screening + defect type classification
                </div>

                <br>

                <div>
                    This mode uses the cleaned WeldRight industrial
                    dataset for experimental screening of welding
                    defects from Current, Voltage, Flow, Humidity,
                    Temperature, and Job Temperature.
                </div>

            </div>
            """
        )

        st.write("")

        render_html(
            """
            <div class="banner banner-warning">

                <b>Dataset / process limitation:</b>

                The public WeldRight dataset documentation does not
                independently verify a specific welding process for
                this model. Therefore this mode is presented as
                WeldRight-based industrial screening rather than as
                a MIG, TIG, SMAW, or other process-specific model.

            </div>
            """
        )

        st.write("")

        st.markdown(
            "### WeldRight Process Inputs"
        )

        st.caption(
            "Enter the six process/environment variables used by the "
            "trained WeldRight screening model."
        )

        wr1, wr2, wr3 = st.columns(3)

        with wr1:

            weldright_current = st.number_input(
                "Current (A)",
                min_value=0.0,
                max_value=1000.0,
                value=27.85,
                step=0.01,
                key="weldright_current",
            )

        with wr2:

            weldright_voltage = st.number_input(
                "Voltage (V)",
                min_value=0.0,
                max_value=1000.0,
                value=9.122,
                step=0.001,
                key="weldright_voltage",
            )

        with wr3:

            weldright_flow = st.number_input(
                "Flow (LPM)",
                min_value=0.0,
                max_value=1000.0,
                value=3.12,
                step=0.01,
                key="weldright_flow",
            )

        wr4, wr5, wr6 = st.columns(3)

        with wr4:

            weldright_humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=84.0,
                step=0.1,
                key="weldright_humidity",
            )

        with wr5:

            weldright_temperature = st.number_input(
                "Temperature (°C)",
                min_value=-100.0,
                max_value=200.0,
                value=24.0,
                step=0.1,
                key="weldright_temperature",
            )

        with wr6:

            weldright_job_temp = st.number_input(
                "Job Temp (°C)",
                min_value=-100.0,
                max_value=500.0,
                value=30.1,
                step=0.1,
                key="weldright_job_temp",
            )

        st.write("")

        weldright_threshold = st.slider(
            "Defect Screening Threshold",
            min_value=0.10,
            max_value=0.70,
            value=0.50,
            step=0.05,
            key="weldright_threshold",
            help=(
                "The model returns a defect-class score. A score at "
                "or above this threshold is flagged for Stage 2 "
                "defect-type classification. This score is not a "
                "calibrated probability."
            ),
        )

        st.caption(
            "Lower thresholds increase screening sensitivity but can "
            "also produce more false alarms. The threshold is "
            "adjustable because the current experimental model is "
            "not calibrated for a single operating point."
        )

        st.write("")

        weldright_predict_button = st.button(
            "SCREEN FOR WELD DEFECT",
            type="primary",
            use_container_width=True,
            key="weldright_predict",
        )

        if weldright_predict_button:

            try:

                weldright_result = predict_weldright(
                    current=weldright_current,
                    voltage=weldright_voltage,
                    flow=weldright_flow,
                    humidity=weldright_humidity,
                    temperature=weldright_temperature,
                    job_temp=weldright_job_temp,
                    threshold=weldright_threshold,
                )

            except Exception as exc:

                st.error(
                    "The WeldRight screening could not be completed. "
                    f"Error: {exc}"
                )

            else:

                st.divider()

                render_html(
                    """
                    <div class="section-kicker">
                        WELDRIGHT RESULT
                    </div>

                    <div class="result-heading">
                        Industrial Defect Screening
                    </div>
                    """
                )

                defect_score = float(
                    weldright_result["defect_score"]
                )

                threshold_value = float(
                    weldright_result["threshold"]
                )

                defect_flag = bool(
                    weldright_result["defect_flag"]
                )

                if defect_score >= 0.70:
                    score_level = "HIGH"
                    score_status = "Strong AI indication"

                elif defect_score >= 0.50:
                    score_level = "MEDIUM"
                    score_status = "Moderate AI indication"

                elif defect_score >= 0.30:
                    score_level = "LOW"
                    score_status = "Weak AI indication"

                else:
                    score_level = "UNCERTAIN"
                    score_status = "Very low AI indication"

                r1, r2, r3, r4 = st.columns(4)

                with r1:

                    summary_card(
                        "DEFECT SCREENING",
                        (
                            "DEFECT FLAGGED"
                            if defect_flag
                            else "NO DEFECT FLAG"
                        ),
                        (
                            f"Threshold: "
                            f"{threshold_value * 100:.0f}%"
                        ),
                        (
                            "HIGH"
                            if defect_flag
                            else "LOW"
                        ),
                    )

                with r2:

                    summary_card(
                        "DEFECT SCORE",
                        f"{defect_score * 100:.1f}%",
                        (
                            f"{score_status}; "
                            "not calibrated probability"
                        ),
                        score_level,
                    )

                with r3:

                    if defect_flag:

                        type_text = (
                            weldright_result[
                                "defect_type"
                            ]
                            or "Defect detected"
                        )

                    else:

                        type_text = "Not classified"

                    summary_card(
                        "DEFECT TYPE",
                        type_text,
                        (
                            "Stage 2 classification"
                            if defect_flag
                            else
                            "Stage 2 runs only when flagged"
                        ),
                    )

                with r4:

                    model_display_name = (
                        display_model_name(
                            weldright_result[
                                "model_type"
                            ]
                        )
                    )

                    summary_card(
                        "MODEL",
                        model_display_name,
                        (
                            "Balanced Random Forest • "
                            "Experimental WeldRight model"
                        ),
                        model_name=True,
                    )

                st.write("")

                if defect_flag:

                    render_html(
                        """
                        <div class="banner banner-danger">

                            <b>
                                The entered condition was flagged
                                by the WeldRight defect-screening
                                model.
                            </b>

                            <br><br>

                            The result should be treated as an
                            experimental screening signal, not as
                            a confirmed physical weld defect.

                        </div>
                        """
                    )

                    st.write("")

                    type_score = (
                        weldright_result[
                            "defect_type_score"
                        ]
                    )

                    if type_score is not None:

                        type_name = html.escape(
                            str(
                                weldright_result[
                                    "defect_type"
                                ]
                            )
                        )

                        render_html(
                            f"""
                            <div class="panel">

                                <div class="context-label">
                                    STAGE 2 DEFECT TYPE
                                </div>

                                <div class="context-value">
                                    {type_name}
                                </div>

                                <br>

                                <div class="context-label">
                                    TYPE MODEL SCORE
                                </div>

                                <div class="context-value">
                                    {float(type_score) * 100:.1f}%
                                </div>

                                <br>

                                <div>
                                    This is the Random Forest class
                                    score for the selected defect
                                    type and is not a calibrated
                                    probability.
                                </div>

                            </div>
                            """
                        )

                    st.write("")

                    st.markdown(
                        "### Engineering Review Guidance"
                    )

                    defect_type = weldright_result[
                        "defect_type"
                    ]

                    if defect_type == "Porosity":

                        render_html(
                            """
                            <div class="recommendation-card">

                                <div class="recommendation-title">
                                    Porosity — process review
                                </div>

                                <div>
                                    Review shielding-gas conditions,
                                    gas-flow stability, surface
                                    cleanliness, contamination,
                                    moisture-related conditions,
                                    and equipment setup before
                                    changing process parameters.
                                </div>

                            </div>
                            """
                        )

                    elif defect_type == "Tungsten Inclusion":

                        render_html(
                            """
                            <div class="recommendation-card">

                                <div class="recommendation-title">
                                    Tungsten Inclusion — process review
                                </div>

                                <div>
                                    Review electrode / tungsten
                                    condition, contamination,
                                    process stability, electrode
                                    handling, and equipment setup.
                                    Confirm the applicable qualified
                                    procedure before changing
                                    process settings.
                                </div>

                            </div>
                            """
                        )

                    else:

                        render_html(
                            """
                            <div class="recommendation-card">

                                <div class="recommendation-title">
                                    Defect condition — engineering review
                                </div>

                                <div>
                                    Review the qualified welding
                                    procedure, equipment condition,
                                    consumables, shielding, and
                                    relevant inspection requirements.
                                </div>

                            </div>
                            """
                        )

                else:

                    render_html(
                        """
                        <div class="banner banner-good">

                            <b>
                                The entered condition was not flagged
                                at the selected screening threshold.
                            </b>

                            <br><br>

                            This means the current model score is
                            below the selected screening threshold.
                            It does not prove that the weld will be
                            defect-free.

                        </div>
                        """
                    )

                st.write("")

                feature_importance = (
                    weldright_result[
                        "feature_importance"
                    ]
                )

                if feature_importance:

                    st.markdown(
                        "### Model Feature Importance"
                    )

                    importance_items = sorted(
                        feature_importance.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )

                    importance_html = ""

                    for (
                        feature_name,
                        importance,
                    ) in importance_items:

                        importance_html += (
                            '<div class="change-row">'

                            '<div class="change-name">'
                            f'{html.escape(str(feature_name))}'
                            "</div>"

                            '<div class="change-values">'
                            f'{float(importance) * 100:.1f}%'
                            "</div>"

                            "</div>"
                        )

                    render_html(
                        f"""
                        <div class="recommendation-card">
                            {importance_html}
                        </div>
                        """
                    )

                st.write("")

                render_html(
                    """
                    <div class="banner banner-warning">

                        <b>
                            Engineering limitation:
                        </b>

                        The WeldRight model is an experimental
                        decision-support model trained from historical
                        industrial data.

                        Its scores are not calibrated probabilities
                        and its screening result does not confirm a
                        physical weld defect.

                        <br><br>

                        The current version provides engineering review
                        guidance rather than claiming that increasing
                        or decreasing a specific parameter will
                        causally eliminate the defect.

                        <br><br>

                        Final acceptance should use the applicable WPS,
                        qualified engineering judgement, and appropriate
                        weld inspection / NDT where required.

                    </div>
                    """
                )

                st.write("")

                with st.expander(
                    "How was this WeldRight result generated?"
                ):

                    st.markdown(
                        f"""
                        **Mode:** WeldRight Industrial Screening

                        **Stage 1 model:** Balanced Random Forest binary defect classifier.

                        **Stage 1 score:** {defect_score * 100:.1f}%.

                        **Screening threshold:** {threshold_value * 100:.0f}%.

                        **Stage 1 result:** {"Defect flagged" if defect_flag else "No defect flag"}.

                        **Stage 2:** Porosity vs Tungsten Inclusion classification when Stage 1 is flagged.

                        **Inputs:** Current, Voltage, Flow, Humidity, Temperature, and Job Temperature.

                        **Important:** The model score is not a calibrated probability. The result is experimental decision support and should not be treated as a guarantee of weld quality.
                        """
                    )

        # Keep the existing Pre-Weld / website footer visible
        # while preventing the old WSteel parameter block from
        # running for the WeldRight mode.

        st.divider()

        render_html(
            """
            <div style="
                text-align:center;
                color:#64748b;
                font-size:12px;
                line-height:1.7;
                padding:10px 0;
            ">
                <b style="color:#94a3b8;">
                    Material Integrity AI
                </b>
                <br>
                AI-Assisted Welding Integrity Assessment •
                Engineering Decision Support Prototype
            </div>
            """
        )

        st.stop()


    # ========================================================
    # EXISTING WSTEEL PROCESS MODELS
    # ========================================================

    process_config = get_process_config(
        selected_process
    )

    render_html(
        f"""
        <div class="panel">

            <div class="context-label">
                SELECTED PROCESS
            </div>

            <div class="context-value">
                {html.escape(selected_process)}
            </div>

            <br>

            <div class="context-label">
                PREDICTION TYPE
            </div>

            <div class="context-value">
                {html.escape(process_config["prediction_type"])}
            </div>

            <br>

            <div>
                {html.escape(process_config["description"])}
            </div>

        </div>
        """
    )

    st.write("")

    st.markdown(
        "### Process Parameters"
    )

    st.caption(
        "Units are shown directly beside every parameter. "
        "Allowed ranges are also available through the input help."
    )

    parameters = process_config[
        "parameters"
    ]

    parameter_values = {}

    columns = st.columns(3)

    for index, parameter in enumerate(
        parameters
    ):

        column = columns[
            index % 3
        ]

        with column:

            visible_label = (
                f'{parameter["label"]} '
                f'({parameter["unit"]})'
            )

            parameter_values[
                parameter["key"]
            ] = st.number_input(
                visible_label,
                min_value=float(
                    parameter["min"]
                ),
                max_value=float(
                    parameter["max"]
                ),
                value=float(
                    parameter["default"]
                ),
                step=float(
                    parameter["step"]
                ),
                key=(
                    f"pre_"
                    f"{selected_process}_"
                    f"{parameter['key']}"
                ),
                help=(
                    f'Allowed range: '
                    f'{parameter["min"]} to '
                    f'{parameter["max"]} '
                    f'{parameter["unit"]}'
                ),
            )

    if selected_process == "Laser Welding":

        render_html(
            """
            <div class="banner banner-warning">

                <b>Current WSteel model note:</b>

                The material thickness / strength field is kept
                constant by the recommendation engine and is not
                treated as a controllable process adjustment.

            </div>
            """
        )

    st.write("")

    predict_button = st.button(
        "PREDICT DEFECT RISK",
        type="primary",
        use_container_width=True,
        disabled=(
            not process_config[
                "model_available"
            ]
        ),
    )

    if not process_config[
        "model_available"
    ]:

        render_html(
            """
            <div class="banner banner-warning">

                <b>
                    No trained model is connected
                    to this welding process yet.
                </b>

                <br><br>

                The process-specific input interface is
                ready, but an AI prediction should only
                be activated after a suitable defect-labelled
                dataset is prepared and the dedicated model
                is trained and evaluated.

            </div>
            """
        )

    if predict_button:

        if selected_process != "Laser Welding":

            st.error(
                "No trained model is available for this process."
            )

        else:

            try:

                result = predict_crack_risk(
                    power=parameter_values[
                        "power"
                    ],
                    welding_speed=parameter_values[
                        "welding_speed"
                    ],
                    gas_flow_rate=parameter_values[
                        "gas_flow_rate"
                    ],
                    focal_position=parameter_values[
                        "focal_position"
                    ],
                    angular_position=parameter_values[
                        "angular_position"
                    ],
                    material_strength=parameter_values[
                        "material_strength"
                    ],
                )

                recommendation_result = (
                    recommend_parameter_changes(
                        power=parameter_values[
                            "power"
                        ],
                        welding_speed=parameter_values[
                            "welding_speed"
                        ],
                        gas_flow_rate=parameter_values[
                            "gas_flow_rate"
                        ],
                        focal_position=parameter_values[
                            "focal_position"
                        ],
                        angular_position=parameter_values[
                            "angular_position"
                        ],
                        material_strength=parameter_values[
                            "material_strength"
                        ],
                        top_n=3,
                    )
                )

            except Exception as exc:

                st.error(
                    "The pre-weld prediction could not "
                    "be completed. "
                    f"Error: {exc}"
                )

            else:

                st.divider()

                render_html(
                    """
                    <div class="section-kicker">
                        PRE-WELD RESULT
                    </div>

                    <div class="result-heading">
                        Defect-Risk Assessment
                    </div>
                    """
                )

                crack_score = (
                    result[
                        "crack_risk_score"
                    ]
                )

                if crack_score >= 0.70:
                    risk_level = "HIGH"

                elif crack_score >= 0.30:
                    risk_level = "MEDIUM"

                else:
                    risk_level = "LOW"

                observed_rate = (
                    recommendation_result[
                        "current_observed_crack_rate"
                    ]
                )

                observed_samples = (
                    recommendation_result[
                        "current_observed_samples"
                    ]
                )

                r1, r2, r3, r4 = st.columns(4)

                with r1:

                    summary_card(
                        "PREDICTED DEFECT",
                        (
                            "Crack risk detected"
                            if result[
                                "prediction"
                            ] == 1
                            else
                            "No crack risk detected"
                        ),
                        "Target: cracking in the weld metal",
                        risk_level,
                    )

                with r2:

                    summary_card(
                        "MODEL SCORE",
                        f"{crack_score * 100:.1f}%",
                        (
                            "Random Forest class-1 score; "
                            "not calibrated probability"
                        ),
                        risk_level,
                    )

                with r3:

                    if observed_rate is None:

                        observed_text = (
                            "Not observed exactly"
                        )

                        observed_sub = (
                            "No exact WSteel condition match"
                        )

                    else:

                        observed_text = (
                            f"{observed_rate * 100:.1f}%"
                        )

                        observed_sub = (
                            "Observed WSteel rate, "
                            f"n={observed_samples}"
                        )

                    summary_card(
                        "EXPERIMENTAL CRACK RATE",
                        observed_text,
                        observed_sub,
                    )

                with r4:

                    model_display_name = (
                        display_model_name(
                            result["model_type"]
                        )
                    )

                    summary_card(
                        "MODEL",
                        model_display_name,
                        (
                            "Random Forest Classifier • "
                            "Experimental WSteel model"
                        ),
                        model_name=True,
                    )

                st.write("")

                if result[
                    "prediction"
                ] == 1:

                    render_html(
                        """
                        <div class="banner banner-danger">

                            <b>
                                Crack risk detected for
                                the entered process condition.
                            </b>

                            <br><br>

                            Review the suggested lower-crack
                            observed alternatives below before
                            selecting welding parameters.

                        </div>
                        """
                    )

                else:

                    render_html(
                        """
                        <div class="banner banner-good">

                            <b>
                                No crack risk was detected
                                by the current model.
                            </b>

                            This is an AI prediction and does
                            not replace weld-quality verification.

                        </div>
                        """
                    )

                st.write("")

                st.markdown(
                    "### Parameter Recommendations"
                )

                recommendations = (
                    recommendation_result[
                        "recommendations"
                    ]
                )

                if recommendations:

                    st.caption(
                        "These alternatives are restricted "
                        "to welding conditions actually observed "
                        "in the WSteel dataset and keep the "
                        "entered material field unchanged."
                    )

                    for index, recommendation in enumerate(
                        recommendations,
                        start=1,
                    ):

                        changes_html = ""

                        for change in recommendation[
                            "changes"
                        ]:

                            current_value = float(
                                change["current"]
                            )

                            recommended_value = float(
                                change["recommended"]
                            )

                            unit = html.escape(
                                str(
                                    change["unit"]
                                )
                            )

                            label = html.escape(
                                str(
                                    change["label"]
                                )
                            )

                            changes_html += (
                                '<div class="change-row">'

                                '<div class="change-name">'
                                f'{label}'
                                "</div>"

                                '<div class="change-values">'
                                f'{current_value:g} '
                                f'{unit}'
                                " → "
                                f'<b>{recommended_value:g} '
                                f'{unit}</b>'
                                "</div>"

                                "</div>"
                            )

                        observed_rate_value = float(
                            recommendation[
                                "observed_crack_rate"
                            ]
                        )

                        model_score_value = float(
                            recommendation[
                                "predicted_risk"
                            ]
                        )

                        model_score_pct = (
                            model_score_value * 100.0
                        )

                        observed_rate_pct = (
                            observed_rate_value * 100.0
                        )

                        change_type = html.escape(
                            str(
                                recommendation[
                                    "change_type"
                                ]
                            )
                        )

                        sample_count = int(
                            recommendation[
                                "observed_samples"
                            ]
                        )

                        render_html(
                            f"""
                            <div class="recommendation-card">

                                <div class="recommendation-title">
                                    Recommendation {index}:
                                    {change_type}
                                </div>

                                <div class="recommendation-metric">

                                    Model score:
                                    <b>
                                        {model_score_pct:.1f}%
                                    </b>

                                    &nbsp;•&nbsp;

                                    Observed crack rate:
                                    <b>
                                        {observed_rate_pct:.1f}%
                                    </b>

                                    &nbsp;•&nbsp;

                                    Experimental samples:
                                    <b>
                                        {sample_count}
                                    </b>

                                </div>

                                <div>
                                    {changes_html}
                                </div>

                            </div>
                            """
                        )

                else:

                    render_html(
                        """
                        <div class="banner banner-warning">

                            <b>
                                No lower-crack observed
                                alternative was found.
                            </b>

                            <br><br>

                            The recommendation engine only
                            considers experimentally observed
                            WSteel conditions, so it does not
                            invent new parameter combinations.

                        </div>
                        """
                    )

                st.write("")

                render_html(
                    """
                    <div class="banner banner-warning">

                        <b>
                            Engineering limitation:
                        </b>

                        These recommendations are
                        model- and dataset-based decision
                        support.

                        They do not prove that changing a
                        parameter causes the defect rate to fall.

                        Confirm every proposed setting against
                        the applicable WPS, equipment limits,
                        material requirements, and qualified
                        engineering judgement before use.

                    </div>
                    """
                )

                st.write("")

                with st.expander(
                    "How was this pre-weld result generated?"
                ):

                    st.markdown(
                        f"""
                        **Process:** {selected_process}

                        **Prediction target:** cracking in the weld metal.

                        **Model:** Random Forest Classifier trained on the cleaned WSteel dataset.

                        **Current model score:** {crack_score * 100:.1f}% for class 1.

                        **Experimental evidence:** The recommendation engine checks the actual WSteel experimental conditions and their observed crack rates.

                        **Recommendation rule:** Only observed conditions with a lower measured crack rate than the current exact condition are considered when an exact current condition exists.

                        **Important:** The model score is not a calibrated probability, and the observed crack rate is not a guarantee of future performance.
                        """
                    )


# ============================================================
# IN-PROCESS PAGE
# ============================================================

elif st.session_state.page == "inprocess":

    if st.button(
        "← Back to Home",
        key="inprocess_home",
    ):
        go_home()

    st.write("")

    render_html(
        """
        <div class="section-kicker">
            STAGE 02 / DURING WELDING
        </div>

        <div class="page-title">
            In-Process Defect Prediction
        </div>

        <div class="page-description">
            AI-assisted monitoring of the welding operation while
            the weld is being produced.
        </div>
        """
    )

    st.write("")

    render_html(
        """
        <div class="banner banner-warning">

            <b>
                In-process model integration is currently pending.
            </b>

            <br><br>

            This section has been intentionally prepared as a
            separate module for the in-process welding defect
            detection model being developed independently.

            <br><br>

            Once the model interface is available, this page can
            connect its inputs, prediction output, defect classes,
            confidence scores and engineering recommendations
            without changing the overall platform structure.

        </div>
        """
    )

    st.write("")

    a, b, c = st.columns(3)

    with a:

        render_html(
            """
            <div class="panel">

                <div class="context-label">
                    STAGE
                </div>

                <div class="context-value">
                    In-Process Welding
                </div>

                <br>

                <div class="context-label">
                    OBJECTIVE
                </div>

                <div class="context-value">
                    Detect potential defects while welding
                </div>

            </div>
            """
        )

    with b:

        render_html(
            """
            <div class="panel">

                <div class="context-label">
                    MODEL STATUS
                </div>

                <div class="context-value">
                    Integration Pending
                </div>

                <br>

                <div class="context-label">
                    MODEL SOURCE
                </div>

                <div class="context-value">
                    Separate project model
                </div>

            </div>
            """
        )

    with c:

        render_html(
            """
            <div class="panel">

                <div class="context-label">
                    FUTURE OUTPUT
                </div>

                <div class="context-value">
                    Defect class + confidence + process response
                </div>

            </div>
            """
        )

    st.write("")

    st.markdown(
        "### Intended In-Process Workflow"
    )

    i1, i2, i3, i4 = st.columns(4)

    with i1:

        render_html(
            """
            <div class="workflow-step">
                <div class="workflow-step-number">
                    01
                </div>
                <div class="workflow-step-title">
                    Welding Data
                </div>
            </div>
            """
        )

    with i2:

        render_html(
            """
            <div class="workflow-step">
                <div class="workflow-step-number">
                    02
                </div>
                <div class="workflow-step-title">
                    AI Detection
                </div>
            </div>
            """
        )

    with i3:

        render_html(
            """
            <div class="workflow-step">
                <div class="workflow-step-number">
                    03
                </div>
                <div class="workflow-step-title">
                    Defect Risk
                </div>
            </div>
            """
        )

    with i4:

        render_html(
            """
            <div class="workflow-step">
                <div class="workflow-step-number">
                    04
                </div>
                <div class="workflow-step-title">
                    Corrective Response
                </div>
            </div>
            """
        )

    st.write("")
    st.write("")

    render_html(
        """
        <div class="panel">

            <div class="context-label">
                INTEGRATION NOTE
            </div>

            <div style="
                color:#aebbd0;
                line-height:1.8;
            ">
                The final implementation of this page should use
                the actual inputs and outputs of the separate
                in-process model. No synthetic prediction logic
                has been added here.
            </div>

        </div>
        """
    )


# ============================================================
# POST-WELD PAGE
# ============================================================

elif st.session_state.page == "post":

    if st.button(
        "← Back to Home",
        key="post_home",
    ):
        go_home()

    st.write("")

    render_html(
        """
        <div class="section-kicker">
            STAGE 03 / AFTER WELDING
        </div>

        <div class="page-title">
            Post-Weld Defect Detection
        </div>

        <div class="page-description">
            Upload a completed weld image and use the trained
            YOLO-based visual inspection model to identify visible
            defect indications and generate preliminary engineering
            guidance.
        </div>
        """
    )

    st.write("")

    left, right = st.columns(
        [0.9, 1.25],
        gap="large",
    )

    with left:

        st.markdown(
            "### Component Information"
        )

        material = st.selectbox(
            "Material",
            [
                "Carbon Steel",
                "Stainless Steel",
                "Aluminium",
                "Unknown",
            ],
            key="post_material",
        )

        component = st.selectbox(
            "Component Type",
            [
                "Plate",
                "Pipe",
                "Beam / Structural Member",
                "Pressure Component",
                "Other",
            ],
            key="post_component",
        )

        joint = st.selectbox(
            "Weld / Joint Type",
            [
                "Butt Weld",
                "Fillet Weld",
                "Lap Weld",
                "T-Joint",
                "Unknown",
            ],
            key="post_joint",
        )

        loading = st.selectbox(
            "Service Loading",
            [
                "Static",
                "Cyclic / Fatigue",
                "Unknown",
            ],
            key="post_loading",
        )

        input_a, input_b = st.columns(2)

        with input_a:

            stress = st.number_input(
                "Applied Stress (MPa)",
                min_value=0.0,
                max_value=1500.0,
                value=100.0,
                step=10.0,
                key="post_stress",
            )

        with input_b:

            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-100.0,
                max_value=1200.0,
                value=25.0,
                step=5.0,
                key="post_temperature",
            )

    with right:

        st.markdown(
            "### Inspection Image"
        )

        uploaded_file = st.file_uploader(
            "Upload JPG, JPEG or PNG",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            key="post_image",
        )

        image = None

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption="Image ready for analysis",
                use_container_width=True,
            )

    st.write("")

    threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.20,
        step=0.05,
        key="post_threshold",
    )

    run = st.button(
        "RUN AI INSPECTION",
        type="primary",
        use_container_width=True,
        key="post_run",
    )

    if run:

        if image is None:

            st.error(
                "Please upload a weld image first."
            )

        else:

            with st.spinner(
                "Analysing weld image..."
            ):

                image_array = np.array(image)

                raw_output = run_post_weld_model(
                    image_array,
                    conf=0.01,
                    iou=0.45,
                )

                detections = raw_output["defects"]

            detections.sort(
                key=lambda item: item["confidence"],
                reverse=True,
            )

            if not detections:

                st.divider()

                render_html(
                    """
                    <div class="section-kicker">
                        INSPECTION RESULT
                    </div>

                    <div class="result-heading">
                        No Defect Candidate Detected
                    </div>

                    <div class="banner banner-warning">

                        <b>
                            No visual defect candidate was returned
                            by the model.
                        </b>

                        <br><br>

                        This does not prove that the weld is
                        defect-free.

                        Manual inspection or appropriate NDT
                        may still be required.

                    </div>
                    """
                )

            else:

                best = detections[0]

                defect = best["name"]
                confidence = best["confidence"]

                if confidence < threshold:

                    st.divider()

                    render_html(
                        """
                        <div class="section-kicker">
                            INSPECTION RESULT
                        </div>

                        <div class="result-heading">
                            No Confident Defect Detection
                        </div>
                        """
                    )

                    a, b, c = st.columns(3)

                    with a:

                        summary_card(
                            "HIGHEST AI CANDIDATE",
                            defect,
                            "Low-confidence model tendency",
                        )

                    with b:

                        summary_card(
                            "CANDIDATE CONFIDENCE",
                            f"{confidence * 100:.1f}%",
                            "Below selected threshold",
                        )

                    with c:

                        summary_card(
                            "STATUS",
                            "UNVERIFIED",
                            "Manual verification required",
                            "UNCERTAIN",
                        )

                    st.write("")

                    render_html(
                        f"""
                        <div class="banner banner-warning">

                            No prediction exceeded the selected
                            <b>{threshold * 100:.0f}%</b>
                            confidence threshold.

                            <br><br>

                            The strongest candidate was
                            <b>{html.escape(defect)}</b>
                            at
                            <b>{confidence * 100:.1f}%</b>.

                            <br><br>

                            Treat this as an AI indication only.

                        </div>
                        """
                    )

                else:

                    confidence_level, confidence_text = (
                        confidence_status(
                            confidence
                        )
                    )

                    data = DEFECT_DB.get(
                        defect,
                        {
                            "severity": "MEDIUM",
                            "causes": [
                                "Further engineering assessment required"
                            ],
                            "verification": [
                                "Qualified inspection required"
                            ],
                            "recommendations": [
                                "Further inspection recommended"
                            ],
                        },
                    )

                    severity = calculate_severity(
                        data["severity"],
                        defect,
                        loading,
                        stress,
                        temperature,
                    )

                    risk = preliminary_risk(
                        severity,
                        confidence,
                    )

                    with st.spinner(
                        "Preparing accepted detection overlay..."
                    ):

                        accepted_output = (
                            run_post_weld_model(
                                image_array,
                                conf=threshold,
                                iou=0.45,
                            )
                        )

                    accepted = accepted_output[
                        "defects"
                    ]

                    overlay = accepted_output[
                        "overlay"
                    ]

                    st.divider()

                    render_html(
                        """
                        <div class="section-kicker">
                            INSPECTION RESULT
                        </div>

                        <div class="result-heading">
                            Weld Assessment Summary
                        </div>
                        """
                    )

                    s1, s2, s3, s4 = st.columns(4)

                    with s1:

                        summary_card(
                            "DETECTED INDICATION",
                            defect,
                            "Highest-confidence AI detection",
                        )

                    with s2:

                        summary_card(
                            "AI CONFIDENCE",
                            f"{confidence * 100:.1f}%",
                            confidence_text,
                        )

                    with s3:

                        summary_card(
                            "PRELIMINARY SEVERITY",
                            severity,
                            "Rule-based interpretation",
                            severity,
                        )

                    with s4:

                        summary_card(
                            "PRELIMINARY RISK",
                            risk,
                            "Not a certified failure prediction",
                            risk,
                        )

                    st.write("")

                    if confidence >= 0.70:

                        render_html(
                            """
                            <div class="banner banner-good">

                                <b>
                                    Strong AI visual indication.
                                </b>

                                The model has relatively high
                                confidence.

                                Engineering verification is still
                                recommended.

                            </div>
                            """
                        )

                    elif confidence >= 0.30:

                        render_html(
                            """
                            <div class="banner banner-warning">

                                <b>
                                    Moderate / low-confidence
                                    AI detection.
                                </b>

                                Confirm this indication manually
                                or using suitable NDT.

                            </div>
                            """
                        )

                    else:

                        render_html(
                            """
                            <div class="banner banner-danger">

                                <b>
                                    Low-confidence AI detection.
                                </b>

                                Do not make an engineering decision
                                from this result alone.

                            </div>
                            """
                        )

                    st.write("")

                    st.markdown(
                        "### Visual Inspection"
                    )

                    im1, im2 = st.columns(2)

                    with im1:

                        st.markdown(
                            "**Original Image**"
                        )

                        st.image(
                            image,
                            use_container_width=True,
                        )

                    with im2:

                        st.markdown(
                            "**AI Detection Overlay**"
                        )

                        st.image(
                            overlay,
                            use_container_width=True,
                        )

                    st.write("")

                    st.markdown(
                        "### Detected Indications"
                    )

                    accepted.sort(
                        key=lambda item: item[
                            "confidence"
                        ],
                        reverse=True,
                    )

                    rows = ""

                    for item in accepted[:10]:

                        rows += (
                            '<div class="detection-row">'
                            '<div class="detection-name">'
                            f'{html.escape(item["name"])}'
                            "</div>"
                            '<div class="detection-confidence">'
                            f'{item["confidence"] * 100:.1f}%'
                            "</div>"
                            "</div>"
                        )

                    if not rows:

                        rows = (
                            '<div class="detection-row">'
                            '<div class="detection-name">'
                            "No detection exceeded the "
                            "selected threshold."
                            "</div>"
                            '<div class="detection-confidence">'
                            "—"
                            "</div>"
                            "</div>"
                        )

                    render_html(
                        f"""
                        <div class="panel">
                            {rows}
                        </div>
                        """
                    )

                    st.write("")

                    st.markdown(
                        "### Component Context"
                    )

                    p1, p2, p3 = st.columns(3)

                    with p1:

                        render_html(
                            f"""
                            <div class="panel">

                                <div class="context-label">
                                    MATERIAL
                                </div>

                                <div class="context-value">
                                    {html.escape(material)}
                                </div>

                                <br>

                                <div class="context-label">
                                    COMPONENT
                                </div>

                                <div class="context-value">
                                    {html.escape(component)}
                                </div>

                            </div>
                            """
                        )

                    with p2:

                        render_html(
                            f"""
                            <div class="panel">

                                <div class="context-label">
                                    JOINT TYPE
                                </div>

                                <div class="context-value">
                                    {html.escape(joint)}
                                </div>

                                <br>

                                <div class="context-label">
                                    LOADING
                                </div>

                                <div class="context-value">
                                    {html.escape(loading)}
                                </div>

                            </div>
                            """
                        )

                    with p3:

                        render_html(
                            f"""
                            <div class="panel">

                                <div class="context-label">
                                    APPLIED STRESS
                                </div>

                                <div class="context-value">
                                    {stress:.1f} MPa
                                </div>

                                <br>

                                <div class="context-label">
                                    TEMPERATURE
                                </div>

                                <div class="context-value">
                                    {temperature:.1f} °C
                                </div>

                            </div>
                            """
                        )

                    st.write("")

                    st.markdown(
                        "### Probable Root-Cause Factors"
                    )

                    causes = list(
                        data["causes"]
                    )

                    if (
                        defect == "Surface Crack-SC"
                        and loading == "Cyclic / Fatigue"
                    ):

                        causes.append(
                            "Cyclic loading may increase fatigue-related crack concern"
                        )

                    if stress >= 300:

                        causes.append(
                            "High service stress may increase defect significance"
                        )

                    if temperature >= 400:

                        causes.append(
                            "Elevated temperature may influence weld behaviour"
                        )

                    root_html = "<br>".join(
                        f"• {html.escape(item)}"
                        for item in causes
                    )

                    render_html(
                        f"""
                        <div class="panel">
                            {root_html}
                        </div>
                        """
                    )

                    st.caption(
                        "These are probable associated factors, "
                        "not proof of the exact cause."
                    )

                    st.write("")

                    st.markdown(
                        "### Recommended Verification"
                    )

                    verification_html = "<br>".join(
                        f"• {html.escape(item)}"
                        for item in data[
                            "verification"
                        ]
                    )

                    render_html(
                        f"""
                        <div class="panel">
                            {verification_html}
                        </div>
                        """
                    )

                    st.write("")

                    st.markdown(
                        "### Engineering Recommendation"
                    )

                    recommendations = list(
                        data["recommendations"]
                    )

                    if (
                        loading == "Cyclic / Fatigue"
                        and severity in [
                            "HIGH",
                            "CRITICAL",
                        ]
                    ):

                        recommendations.append(
                            "Consider fatigue-related engineering assessment before continued cyclic operation"
                        )

                    recommendation_html = "<br>".join(
                        f"• {html.escape(item)}"
                        for item in recommendations
                    )

                    render_html(
                        f"""
                        <div class="panel">
                            {recommendation_html}
                        </div>
                        """
                    )

                    st.write("")

                    with st.expander(
                        "How was this result generated?"
                    ):

                        st.markdown(
                            """
                            **Defect identification**  
                            Generated by the trained YOLO weld-defect model.

                            **AI confidence**  
                            Direct YOLO detection score; it is not a calibrated probability.

                            **Preliminary severity**  
                            Generated using rule-based engineering logic.

                            **Root-cause factors**  
                            Generated from the project's engineering knowledge base.

                            **Preliminary risk**  
                            Qualitative decision-support output, not a trained failure-prediction model.

                            **Verification**  
                            Engineering decisions should follow applicable inspection standards and qualified assessment.
                            """
                        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

render_html(
    """
    <div style="
        text-align:center;
        color:#64748b;
        font-size:12px;
        line-height:1.7;
        padding:10px 0;
    ">
        <b style="color:#94a3b8;">
            Material Integrity AI
        </b>
        <br>
        AI-Assisted Welding Integrity Assessment •
        Engineering Decision Support Prototype
    </div>
    """
)