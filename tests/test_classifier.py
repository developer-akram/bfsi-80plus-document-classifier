from src.classifier import (
    calculate_bbox_iou,
    classify_bfsi_document,
    fuzzy_kyc_name_similarity,
)


def test_cheque_vs_demand_draft_classification():
    cheque_text = "PAY TO AKRAM KHAN OR BEARER RUPEES FIFTY THOUSAND IFS SBIN0004321"
    dd_text = "ON DEMAND PAY TO XYZ ENTERPRISES DEMAND DRAFT ISSUING BRANCH MUMBAI"

    cls_cheque, score_c = classify_bfsi_document(cheque_text, {"MICR_BOTTOM_BAND"})
    cls_dd, score_d = classify_bfsi_document(dd_text, {"MICR_BOTTOM_BAND"})

    assert cls_cheque == "CHEQUE" and score_c >= 0.9
    assert cls_dd == "DEMAND_DRAFT" and score_d >= 0.9


def test_aof_and_loan_classification():
    aof_text = "SAVINGS ACCOUNT OPENING FORM NOMINEE DETAILS DECLARATION CUSTOMER SIGNATURE"
    cls_aof, score = classify_bfsi_document(aof_text, {"CHECKBOX_GRID"})
    assert cls_aof == "AOF_FORM" and score >= 0.9


def test_bbox_iou_and_fuzzy_name_match():
    assert calculate_bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert calculate_bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    sim = fuzzy_kyc_name_similarity("MOHAMMED AKRAM KHAN", "MOHAMMAD AKRAM KHAN")
    assert sim >= 0.94
