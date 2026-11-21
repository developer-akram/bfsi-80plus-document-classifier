"""Config-driven 80+ class BFSI document taxonomy classifier, IoU evaluator, and DP fuzzy matcher."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentRule:
    doc_class: str
    category: str
    weighted_anchors: dict[str, float]
    regex_patterns: tuple[str, ...] = ()
    layout_hints: tuple[str, ...] = ()


BFSI_TAXONOMY_SAMPLE: tuple[DocumentRule, ...] = (
    DocumentRule(
        doc_class="CHEQUE",
        category="BANKING_INSTRUMENT",
        weighted_anchors={"PAY": 0.25, "RUPEES": 0.25, "OR BEARER": 0.25, "IFS": 0.25},
        regex_patterns=(r"\b[A-Z]{4}0[A-Z0-9]{6}\b",),
        layout_hints=("MICR_BOTTOM_BAND",),
    ),
    DocumentRule(
        doc_class="DEMAND_DRAFT",
        category="BANKING_INSTRUMENT",
        weighted_anchors={"ON DEMAND": 0.35, "DEMAND DRAFT": 0.35, "ISSUING BRANCH": 0.15, "DRAWEE": 0.15},
        layout_hints=("MICR_BOTTOM_BAND",),
    ),
    DocumentRule(
        doc_class="AOF_FORM",
        category="ACCOUNT_OPENING",
        weighted_anchors={"ACCOUNT OPENING": 0.35, "NOMINEE": 0.25, "DECLARATION": 0.2, "CUSTOMER SIGNATURE": 0.2},
        layout_hints=("CHECKBOX_GRID",),
    ),
    DocumentRule(
        doc_class="LOAN_APPLICATION_PACKET",
        category="LOAN_ORIGINATION",
        weighted_anchors={"LOAN APPLICATION": 0.35, "SANCTION": 0.2, "TENURE": 0.2, "EMI": 0.25},
        layout_hints=("CHECKBOX_GRID", "MULTI_PAGE_CONTINUATION"),
    ),
    DocumentRule(
        doc_class="AADHAAR_CARD",
        category="KYC_IDENTITY",
        weighted_anchors={"GOVERNMENT OF INDIA": 0.4, "UNIQUE IDENTIFICATION": 0.3, "DOB": 0.15, "MALE": 0.15},
        regex_patterns=(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b",),
    ),
)


def classify_bfsi_document(ocr_text: str, layout_cues: set[str] | None = None) -> tuple[str, float]:
    """Score extracted OCR text and layout cues against the declarative taxonomy."""
    upper_text = ocr_text.upper()
    cues = layout_cues or set()
    best_class, best_score = "UNKNOWN", 0.0

    for rule in BFSI_TAXONOMY_SAMPLE:
        score = 0.0
        for anchor, weight in rule.weighted_anchors.items():
            if anchor in upper_text:
                score += weight
        for pattern in rule.regex_patterns:
            if re.search(pattern, upper_text):
                score += 0.15
        for hint in rule.layout_hints:
            if hint in cues:
                score += 0.10
        normalized = round(min(1.0, score), 4)
        if normalized > best_score:
            best_class, best_score = rule.doc_class, normalized

    return best_class, best_score


def calculate_bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union (IoU) for [x_min, y_min, x_max, y_max] boxes."""
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return round(inter / union, 4) if union > 0 else 0.0


def fuzzy_kyc_name_similarity(pan_name: str, aadhaar_name: str) -> float:
    """Dynamic Programming Levenshtein similarity ratio for KYC name verification."""
    a, b = " ".join(pan_name.upper().split()), " ".join(aadhaar_name.upper().split())
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return round(1.0 - (dp[m][n] / max(m, n, 1)), 4)
