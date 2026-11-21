# BFSI 80+ Document Classification & Visual Verification Toolkit

A clean, config-driven **Document Intelligence & Visual Matching** reference architecture designed for multi-class BFSI packets (KYC Identity/Address Proofs, Account Opening Forms [AOF], Loan Origination Packets, Cheques, Demand Drafts [DD], and Bank Statements):

- **Declarative 80+ Document Taxonomy Engine:** Distinguishes visually similar banking instruments (e.g., **Cheque** vs. **Demand Draft [DD]** vs. **Account Opening Form [AOF]**) using weighted anchor scoring, regex signals, and spatial layout zones without brittle `if-else` monoliths.
- **Bounding-Box IoU (Intersection over Union) Evaluator:** Quantifies localization accuracy for signature, stamp, face, and checkbox ROIs.
- **DP Levenshtein Fuzzy KYC Name Matcher:** Computes character-level edit distance and similarity scores to reconcile minor OCR transliteration variants across PAN vs. Aadhaar names.

```bash
pip install -r requirements.txt
pytest -v
```
