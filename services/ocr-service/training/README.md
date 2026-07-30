# pix2tex Fine-Tuning Workflow

This folder prepares a handwritten algebra dataset for pix2tex / LaTeX-OCR fine-tuning.

The goal is to improve Module 01 OCR output quality for realistic handwritten algebra answers.
Module 02 only needs the final JSON output, so training is only for improving OCR accuracy.

## Workflow

1. Run the normal API/pipeline so cropped step images are saved in:

```text
data/regions/
```

2. Prepare a labeling CSV:

```powershell
python training/scripts/prepare_labeling_dataset.py
```

This creates:

```text
training/data/images/
training/data/labels.csv
```

3. Open `training/data/labels.csv` and fill the `latex` column manually.

Example:

```csv
filename,latex,split
sample_0001.png,\frac{2x+3}{(x-1)^2},train
sample_0002.png,2x+3=A(x-1)+B,train
```

4. Validate labels:

```powershell
python training/scripts/validate_labels.py
```

5. Export files for pix2tex:

```powershell
python training/scripts/export_pix2tex_dataset.py
```

This creates:

```text
training/data/pix2tex/train/images/
training/data/pix2tex/train/equations.txt
training/data/pix2tex/val/images/
training/data/pix2tex/val/equations.txt
training/data/pix2tex/test/images/
training/data/pix2tex/test/equations.txt
```

6. Upload the project or exported dataset to Google Colab and follow:

```text
training/colab_finetune_pix2tex.md
```

## Important

Fine-tuning needs paired data:

```text
cropped handwritten expression image -> correct LaTeX label
```

Without labels, pix2tex cannot learn your handwriting style.

