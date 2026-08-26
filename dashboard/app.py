import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import html
import textwrap


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "weld_detector",
    "weld_detector_best.pt"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Material Integrity AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SAFE HTML RENDERER
# Prevents Streamlit from showing <div> as code
# ============================================================

def render_html(content):
    st.html(
        textwrap.dedent(content).strip()
    )

# ============================================================
# STYLING
# ============================================================

render_html(
    """
    <style>

    /* Hide Streamlit's own top header */
    [data-testid="stHeader"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 8% 0%,
                rgba(37, 99, 235, 0.13),
                transparent 30%
            ),
            radial-gradient(
                circle at 92% 0%,
                rgba(14, 165, 233, 0.10),
                transparent 28%
            ),
            #080d14;

        color: #f5f7fb;
    }

    /* Fix clipped header */
    .block-container {
        max-width: 1480px;
        padding-top: 3rem !important;
        padding-bottom: 4rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
    }

    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 850 !important;
    }

    p {
        color: #cbd5e1;
    }

    /* ======================================================
       BRAND
    ====================================================== */

    .brand-title {
        color: #ffffff;
        font-size: clamp(36px, 4vw, 52px);
        font-weight: 950;
        letter-spacing: -1.8px;
        line-height: 1.05;
    }

    .brand-subtitle {
        color: #94a3b8;
        margin-top: 10px;
        font-size: 16px;
        font-weight: 500;
    }

    .online-pill {
        display: inline-block;

        padding: 9px 15px;

        border-radius: 999px;

        background: rgba(34, 197, 94, 0.09);
        border: 1px solid rgba(34, 197, 94, 0.35);

        color: #6ee7a5;

        font-size: 12px;
        font-weight: 900;
        letter-spacing: 0.9px;
    }

    /* ======================================================
       SECTION TEXT
    ====================================================== */

    .section-kicker {
        color: #8ba3c7;

        font-size: 12px;
        font-weight: 900;

        letter-spacing: 1.6px;

        margin-bottom: 9px;
    }

    .result-heading {
        color: #ffffff;

        font-size: clamp(30px, 3vw, 42px);
        font-weight: 950;

        line-height: 1.1;

        margin: 4px 0 22px 0;
    }

    /* ======================================================
       RESULT CARDS
    ====================================================== */

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

        min-height: 135px;
    }

    .summary-label {
        color: #91a0b4;

        font-size: 12px;
        font-weight: 900;

        letter-spacing: 1.2px;

        margin-bottom: 13px;
    }

    .summary-value {
        color: #ffffff;

        font-size: 28px;
        font-weight: 950;

        line-height: 1.12;

        word-break: break-word;
    }

    .summary-sub {
        color: #8290a3;

        margin-top: 9px;

        font-size: 13px;
        line-height: 1.35;
    }

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

    /* ======================================================
       BANNERS
    ====================================================== */

    .banner {
        border-radius: 14px;

        padding: 18px 20px;

        font-size: 15px;
        line-height: 1.6;
    }

    .banner-good {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.30);
        color: #bff3d3;
    }

    .banner-warning {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.32);
        color: #ffe0a4;
    }

    .banner-danger {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.32);
        color: #ffc2c7;
    }

    /* ======================================================
       PANELS
    ====================================================== */

    .panel {
        background: #0f1823;

        border: 1px solid #293648;

        border-radius: 16px;

        padding: 21px 23px;

        color: #d9e2ee;

        font-size: 16px;
        line-height: 1.75;
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

    /* ======================================================
       INPUTS
    ====================================================== */

    [data-testid="stWidgetLabel"] p {
        color: #d6deea !important;

        font-size: 15px !important;
        font-weight: 750 !important;
    }

    div[data-baseweb="select"] > div {
        background: #111b27 !important;

        border: 1px solid #334155 !important;

        color: #ffffff !important;

        border-radius: 10px !important;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }

    [data-testid="stNumberInput"] input {
        background: #111b27 !important;

        color: #ffffff !important;

        font-size: 16px !important;

        border-color: #334155 !important;
    }

    [data-testid="stNumberInput"] button {
        background: #111b27 !important;
        color: #ffffff !important;
    }

    [data-testid="stFileUploader"] section {
        background: #111b27 !important;

        border: 1px dashed #3c4b60 !important;

        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        color: #cbd5e1 !important;
    }

    /* ======================================================
       IMAGES
    ====================================================== */

    [data-testid="stImage"] img {
        max-height: 520px;

        object-fit: contain;

        border-radius: 12px;
    }

    /* ======================================================
       BUTTON
    ====================================================== */

    .stButton > button[kind="primary"] {
        width: 100%;

        min-height: 55px;

        border: none !important;

        border-radius: 12px !important;

        background:
            linear-gradient(
                90deg,
                #2563eb,
                #0ea5e9
            ) !important;

        color: white !important;

        font-size: 15px !important;
        font-weight: 900 !important;

        letter-spacing: 0.4px;
    }

    /* ======================================================
       TABS
    ====================================================== */

    button[data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 800 !important;
    }

    /* ======================================================
       RESPONSIVE
    ====================================================== */

    @media (max-width: 800px) {

        .block-container {
            padding-top: 2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .brand-title {
            font-size: 34px !important;
        }

        .summary-value {
            font-size: 23px !important;
        }

    }

    </style>
    """
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


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
            "Excessive root gap or unsuitable joint preparation"
        ],

        "verification": [
            "Perform visual and dimensional inspection",
            "Assess penetration and remaining thickness",
            "Use suitable NDT where required"
        ],

        "recommendations": [
            "Review welding current and heat input",
            "Inspect the complete affected region",
            "Repair or re-weld if acceptance limits are exceeded"
        ]
    },

    "Contamination-CN": {
        "severity": "MEDIUM",

        "causes": [
            "Oil, oxide, rust or moisture contamination",
            "Poor surface preparation",
            "Contaminated filler material",
            "Inadequate shielding"
        ],

        "verification": [
            "Clean and visually inspect the region",
            "Review shielding conditions",
            "Inspect nearby weld surface"
        ],

        "recommendations": [
            "Improve surface preparation",
            "Check filler and shielding conditions",
            "Reinspect after corrective action"
        ]
    },

    "Lack of Fusion-LOF": {
        "severity": "HIGH",

        "causes": [
            "Insufficient heat input",
            "Excessive travel speed",
            "Incorrect torch or electrode angle",
            "Poor joint preparation"
        ],

        "verification": [
            "Use UT or RT inspection where applicable",
            "Determine defect location and extent",
            "Review joint preparation"
        ],

        "recommendations": [
            "Review welding procedure parameters",
            "Repair the affected region where required",
            "Reinspect after repair"
        ]
    },

    "Normal Weld-NW": {
        "severity": "LOW",

        "causes": [
            "No major visible surface defect indicated by the current model"
        ],

        "verification": [
            "Perform routine visual inspection",
            "Apply normal NDT requirements when applicable"
        ],

        "recommendations": [
            "Continue standard inspection procedures",
            "Record the inspection result"
        ]
    },

    "Spatter-SR": {
        "severity": "LOW",

        "causes": [
            "Excessive welding current",
            "Incorrect voltage",
            "Unstable welding arc",
            "Shielding instability"
        ],

        "verification": [
            "Inspect after cleaning the surface",
            "Check whether spatter obscures other discontinuities"
        ],

        "recommendations": [
            "Remove excessive spatter",
            "Review current and voltage",
            "Check welding arc stability"
        ]
    },

    "Surface Crack-SC": {
        "severity": "CRITICAL",

        "causes": [
            "Residual welding stresses",
            "Rapid cooling",
            "Hydrogen-assisted cracking",
            "Fatigue or stress concentration"
        ],

        "verification": [
            "Perform suitable NDT",
            "Determine crack depth and extent",
            "Obtain qualified engineering evaluation"
        ],

        "recommendations": [
            "Do not rely only on visual inspection",
            "Assess before continued service",
            "Repair or replace affected region where required"
        ]
    },

    "Undercut-UC": {
        "severity": "MEDIUM",

        "causes": [
            "Excessive welding current",
            "High travel speed",
            "Incorrect torch or electrode angle",
            "Improper welding technique"
        ],

        "verification": [
            "Measure actual undercut depth",
            "Compare with applicable acceptance criteria",
            "Inspect the entire weld length"
        ],

        "recommendations": [
            "Adjust welding current and travel speed",
            "Correct torch or electrode angle",
            "Repair if allowable limits are exceeded"
        ]
    }
}


# ============================================================
# HELPERS
# ============================================================

def level_css(level):

    return {
        "LOW": "low",
        "MEDIUM": "medium",
        "HIGH": "high",
        "CRITICAL": "critical",
        "UNCERTAIN": "uncertain"
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


def calculate_severity(
    base,
    defect,
    loading,
    stress,
    temperature
):

    levels = [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]

    index = levels.index(base)

    if (
        loading == "Cyclic / Fatigue"
        and defect in [
            "Surface Crack-SC",
            "Lack of Fusion-LOF",
            "Undercut-UC"
        ]
    ):
        index += 1

    if stress >= 300:
        index += 1

    if temperature >= 400:
        index += 1

    return levels[min(index, 3)]


def preliminary_risk(severity, confidence):

    if confidence < 0.30:
        return "UNCERTAIN"

    return severity


def summary_card(
    label,
    value,
    subtitle,
    level=None
):

    css = level_css(level)

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
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [5, 1]
)

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
                ● MODEL ONLINE
            </span>
        </div>
        """
    )


st.divider()


# ============================================================
# TABS
# ============================================================

post_tab, pre_tab = st.tabs(
    [
        "🔍 Post-Weld Inspection",
        "⚡ Pre / In-Process Prediction"
    ]
)


# ============================================================
# POST-WELD
# ============================================================

with post_tab:

    st.write("")

    render_html(
        """
        <div class="section-kicker">
            VISUAL WELD ASSESSMENT
        </div>
        """
    )

    st.markdown(
        "## Inspect a Completed Weld"
    )

    st.caption(
        "Upload a weld image and provide operating context. "
        "The AI identifies visible defect patterns and provides "
        "a preliminary engineering interpretation."
    )


    left, right = st.columns(
        [0.9, 1.25],
        gap="large"
    )


    # ========================================================
    # COMPONENT INFORMATION
    # ========================================================

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
                "Unknown"
            ]
        )

        component = st.selectbox(
            "Component Type",
            [
                "Plate",
                "Pipe",
                "Beam / Structural Member",
                "Pressure Component",
                "Other"
            ]
        )

        joint = st.selectbox(
            "Weld / Joint Type",
            [
                "Butt Weld",
                "Fillet Weld",
                "Lap Weld",
                "T-Joint",
                "Unknown"
            ]
        )

        loading = st.selectbox(
            "Service Loading",
            [
                "Static",
                "Cyclic / Fatigue",
                "Unknown"
            ]
        )


        input_a, input_b = st.columns(2)

        with input_a:

            stress = st.number_input(
                "Applied Stress (MPa)",
                min_value=0.0,
                max_value=1500.0,
                value=100.0,
                step=10.0
            )

        with input_b:

            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-100.0,
                max_value=1200.0,
                value=25.0,
                step=5.0
            )


    # ========================================================
    # IMAGE
    # ========================================================

    with right:

        st.markdown(
            "### Inspection Image"
        )

        uploaded_file = st.file_uploader(
            "Upload JPG, JPEG or PNG",
            type=["jpg", "jpeg", "png"]
        )

        image = None

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption="Image ready for analysis",
                use_container_width=True
            )


    st.write("")

    threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.20,
        step=0.05
    )


    run = st.button(
        "RUN AI INSPECTION",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # RUN INSPECTION
    # ========================================================

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

                raw_result = model.predict(
                    source=image_array,
                    conf=0.01,
                    iou=0.45,
                    verbose=False
                )[0]


            detections = []

            for box in raw_result.boxes:

                cls = int(
                    box.cls[0]
                )

                conf = float(
                    box.conf[0]
                )

                detections.append(
                    {
                        "name": raw_result.names[cls],
                        "confidence": conf
                    }
                )


            detections.sort(
                key=lambda item: item["confidence"],
                reverse=True
            )


            # ====================================================
            # NOTHING FOUND
            # ====================================================

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
                        <b>No visual defect candidate was returned by the model.</b>
                        <br><br>
                        This does not prove that the weld is defect-free.
                        Manual inspection or appropriate NDT may still be required.
                    </div>
                    """
                )


            else:

                best = detections[0]

                defect = best["name"]
                confidence = best["confidence"]


                # =================================================
                # BELOW ACCEPTANCE THRESHOLD
                # =================================================

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
                            "Low-confidence model tendency"
                        )

                    with b:

                        summary_card(
                            "CANDIDATE CONFIDENCE",
                            f"{confidence * 100:.1f}%",
                            "Below selected threshold"
                        )

                    with c:

                        summary_card(
                            "STATUS",
                            "UNVERIFIED",
                            "Manual verification required",
                            "UNCERTAIN"
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


                # =================================================
                # ACCEPTED DETECTION
                # =================================================

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
                            ]
                        }
                    )


                    severity = calculate_severity(
                        data["severity"],
                        defect,
                        loading,
                        stress,
                        temperature
                    )


                    risk = preliminary_risk(
                        severity,
                        confidence
                    )


                    accepted_result = model.predict(
                        source=image_array,
                        conf=threshold,
                        iou=0.45,
                        verbose=False
                    )[0]


                    overlay = accepted_result.plot()

                    overlay = overlay[
                        :, :, ::-1
                    ]


                    # =================================================
                    # RESULT HEADER
                    # =================================================

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


                    # =================================================
                    # SUMMARY CARDS
                    # =================================================

                    s1, s2, s3, s4 = st.columns(4)


                    with s1:

                        summary_card(
                            "DETECTED INDICATION",
                            defect,
                            "Highest-confidence AI detection"
                        )


                    with s2:

                        summary_card(
                            "AI CONFIDENCE",
                            f"{confidence * 100:.1f}%",
                            confidence_text
                        )


                    with s3:

                        summary_card(
                            "PRELIMINARY SEVERITY",
                            severity,
                            "Rule-based interpretation",
                            severity
                        )


                    with s4:

                        summary_card(
                            "PRELIMINARY RISK",
                            risk,
                            "Not a certified failure prediction",
                            risk
                        )


                    st.write("")


                    # =================================================
                    # CONFIDENCE BANNER
                    # =================================================

                    if confidence >= 0.70:

                        render_html(
                            """
                            <div class="banner banner-good">
                                <b>Strong AI visual indication.</b>
                                The model has relatively high confidence.
                                Engineering verification is still recommended.
                            </div>
                            """
                        )

                    elif confidence >= 0.30:

                        render_html(
                            """
                            <div class="banner banner-warning">
                                <b>Moderate / low-confidence AI detection.</b>
                                Confirm this indication manually or using suitable NDT.
                            </div>
                            """
                        )

                    else:

                        render_html(
                            """
                            <div class="banner banner-danger">
                                <b>Low-confidence AI detection.</b>
                                Do not make an engineering decision from this result alone.
                            </div>
                            """
                        )


                    # =================================================
                    # IMAGE COMPARISON
                    # =================================================

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
                            use_container_width=True
                        )


                    with im2:

                        st.markdown(
                            "**AI Detection Overlay**"
                        )

                        st.image(
                            overlay,
                            use_container_width=True
                        )


                    # =================================================
                    # DETECTED INDICATIONS
                    # =================================================

                    st.write("")
                    st.markdown(
                        "### Detected Indications"
                    )


                    accepted = [
                        item
                        for item in detections
                        if item["confidence"] >= threshold
                    ]


                    rows = ""

                    for item in accepted[:10]:

                        rows += (
                            '<div class="detection-row">'
                            f'<div class="detection-name">{html.escape(item["name"])}</div>'
                            f'<div class="detection-confidence">{item["confidence"] * 100:.1f}%</div>'
                            '</div>'
                        )


                    render_html(
                        f"""
                        <div class="panel">
                            {rows}
                        </div>
                        """
                    )


                    # =================================================
                    # COMPONENT CONTEXT
                    # =================================================

                    st.write("")
                    st.markdown(
                        "### Component Context"
                    )


                    p1, p2, p3 = st.columns(3)


                    with p1:

                        render_html(
                            f"""
                            <div class="panel">
                                <div class="context-label">MATERIAL</div>
                                <div class="context-value">{html.escape(material)}</div>

                                <br>

                                <div class="context-label">COMPONENT</div>
                                <div class="context-value">{html.escape(component)}</div>
                            </div>
                            """
                        )


                    with p2:

                        render_html(
                            f"""
                            <div class="panel">
                                <div class="context-label">JOINT TYPE</div>
                                <div class="context-value">{html.escape(joint)}</div>

                                <br>

                                <div class="context-label">LOADING</div>
                                <div class="context-value">{html.escape(loading)}</div>
                            </div>
                            """
                        )


                    with p3:

                        render_html(
                            f"""
                            <div class="panel">
                                <div class="context-label">APPLIED STRESS</div>
                                <div class="context-value">{stress:.1f} MPa</div>

                                <br>

                                <div class="context-label">TEMPERATURE</div>
                                <div class="context-value">{temperature:.1f} °C</div>
                            </div>
                            """
                        )


                    # =================================================
                    # ROOT CAUSES
                    # =================================================

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
                        "These are probable associated factors, not proof of the exact cause."
                    )


                    # =================================================
                    # VERIFICATION
                    # =================================================

                    st.write("")
                    st.markdown(
                        "### Recommended Verification"
                    )


                    verification_html = "<br>".join(
                        f"• {html.escape(item)}"
                        for item in data["verification"]
                    )


                    render_html(
                        f"""
                        <div class="panel">
                            {verification_html}
                        </div>
                        """
                    )


                    # =================================================
                    # ENGINEERING RECOMMENDATION
                    # =================================================

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
                            "CRITICAL"
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


                    # =================================================
                    # EXPLANATION
                    # =================================================

                    st.write("")

                    with st.expander(
                        "How was this result generated?"
                    ):

                        st.markdown(
                            """
**Defect identification**  
Generated by your trained YOLO model.

**AI confidence**  
Generated directly by YOLO.

**Preliminary severity**  
Currently generated using rule-based engineering logic.

**Root-cause factors**  
Generated from the project's engineering knowledge base.

**Preliminary risk**  
Currently qualitative and is **not yet a trained failure-prediction model**.

**Recommendations**  
Decision-support guidance only. Final engineering decisions should follow applicable inspection standards and qualified assessment.
"""
                        )


# ============================================================
# PRE / IN-PROCESS
# ============================================================

with pre_tab:

    st.write("")

    render_html(
        """
        <div class="section-kicker">
            PREDICTIVE WELD INTELLIGENCE
        </div>
        """
    )

    st.markdown(
        "## Pre-Weld / In-Process Prediction"
    )

    st.caption(
        "This module will predict defect and distortion risk "
        "using actual welding-process parameters."
    )


    a, b, c = st.columns(3)


    with a:

        st.selectbox(
            "Welding Process",
            [
                "MIG / GMAW",
                "TIG / GTAW",
                "SMAW",
                "Submerged Arc Welding",
                "Unknown"
            ],
            key="pre_process"
        )

        st.number_input(
            "Current (A)",
            0.0,
            1000.0,
            180.0,
            key="pre_current"
        )

        st.number_input(
            "Voltage (V)",
            0.0,
            100.0,
            24.0,
            key="pre_voltage"
        )


    with b:

        st.number_input(
            "Travel Speed (mm/s)",
            0.0,
            100.0,
            6.0,
            key="pre_speed"
        )

        st.number_input(
            "Torch / Electrode Angle (°)",
            0.0,
            90.0,
            45.0,
            key="pre_angle"
        )

        st.number_input(
            "Wire Feed Speed",
            0.0,
            100.0,
            5.0,
            key="pre_wire"
        )


    with c:

        st.number_input(
            "Plate Thickness (mm)",
            0.1,
            100.0,
            6.0,
            key="pre_thickness"
        )

        st.selectbox(
            "Material",
            [
                "Carbon Steel",
                "Stainless Steel",
                "Aluminium"
            ],
            key="pre_material"
        )

        st.selectbox(
            "Joint Type",
            [
                "Butt Joint",
                "T-Joint",
                "Lap Joint",
                "Corner Joint"
            ],
            key="pre_joint"
        )


    st.write("")

    st.button(
        "PREDICT DEFECT & DISTORTION RISK",
        disabled=True,
        use_container_width=True
    )


    render_html(
        """
        <div class="banner banner-warning">
            <b>Predictive ML model not trained yet.</b>
            <br><br>
            This section will be activated after we train the
            pre-weld defect and distortion prediction model.
        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Material Integrity AI • Weld Intelligence Platform • "
    "AI-Assisted Engineering Prototype"
)