
# Layout Parsing Guide for Sydney Notes

This document is your visual and conceptual guide to understanding how we label, train, and use layout detection models for clinical documents. These labels are part of our **ubiquitous language** — shared terms that mean the same thing to humans and machines across every step of the process.

---

## 🧱 Core Components (What They Are)

### 🖼️ `makesense.ai`  
A free browser-based tool where we **draw boxes** and assign labels (like `Header`, `Table`, etc.).  
🟢 Use it to create COCO-style annotations by uploading `.png` images and your class list.

### 📦 `COCO JSON`  
The file format used to store your labeled boxes.  
✅ It tells the model: *"This image has a 'Header' from (x1, y1) to (x2, y2)."*

### 🧠 `Detectron2` or `EfficientDet`  
These are the machine learning models that **learn how to detect layout regions**.  
- `Detectron2` = flexible, very powerful, great for complex layouts  
- `EfficientDet` = lightweight, fast, easier to deploy  
You train one of these models **on the labeled images + COCO JSON.**

---

## 🔄 Workflow Overview

```
PDFs (original)
   │
   └─▶ `pdf_to_images_sydney_notes.py`
          Converts PDF pages to `.png` images
   │
   └─▶ `makesense.ai`
          You draw boxes and label each image region
   │
   └─▶ COCO JSON export
          Structured annotation file with labels and coordinates
   │
   └─▶ Model training (Detectron2 or EfficientDet)
          Learns to detect headers, tables, footers, etc.
   │
   └─▶ Parsing engine
          Applies the trained model to unseen documents
```

---

## ✅ Label Definitions

Use these consistently when annotating documents in `makesense.ai`.

### `Header`
Top banner: clinic name, address, etc.

### `ClientProviderDemographics`
Name, DOB, provider ID, or any top-left metadata.

### `LeftColumnMetadata`
Left-side content like vitals, appointment time, etc.

### `RightColumnNarrative`
Narrative block: “Client discussed...”

### `VisitBody`
Full content of the progress note — SOAP structure or freeform text.

### `Table`
Gridded or bordered tabular content.

### `SignatureBlock`
Attestation, provider signature, signoff area.

### `FooterDocumentDatestamp`
Printed timestamps, EHR footer, or system notes.

### `PageNumber`
“Page 1 of 2” etc., if visually present.

---

## 🧠 Notes on Usage

- Label **everything that is consistent and repeated across documents**
- Each label should describe **how the region is used**, not just where it appears
- Use **only one label per region**
- Don’t mix in text anchor concepts here — they belong in the OCR pipeline

---

## 🔖 What’s Next

Once labeled:
- Export from makesense.ai as **COCO JSON**
- Place it in:  
  `/data/layout_training_sydney_notes/annotations.json`

Then we’ll run the training step — the model will learn from what you drew.

---