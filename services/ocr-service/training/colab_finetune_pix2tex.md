# Google Colab Fine-Tuning Notes

Use Colab because pix2tex training is heavy on CPU.

## 1. Upload Dataset

Run locally first:

```powershell
python training/scripts/prepare_labeling_dataset.py
# Fill training/data/labels.csv
python training/scripts/validate_labels.py
python training/scripts/export_pix2tex_dataset.py
```

Then upload this folder to Colab:

```text
training/data/pix2tex/
```

## 2. Install Training Package

In Colab:

```python
!pip install "pix2tex[train]"
```

## 3. Create pix2tex Dataset Files

pix2tex training uses image folders plus an equation text file. A typical
dataset build command is:

```python
!python -m pix2tex.dataset.dataset \
  --equations training/data/pix2tex/train/equations.txt \
  --images training/data/pix2tex/train/images \
  --out training/data/pix2tex/train.pkl
```

Repeat for validation:

```python
!python -m pix2tex.dataset.dataset \
  --equations training/data/pix2tex/val/equations.txt \
  --images training/data/pix2tex/val/images \
  --out training/data/pix2tex/val.pkl
```

## 4. Fine-Tune

pix2tex training configuration can vary by version. Start from the installed
package's default config, then point it to:

```text
training/data/pix2tex/train.pkl
training/data/pix2tex/val.pkl
training/checkpoints/
```

Typical command pattern:

```python
!python -m pix2tex.train --config training/configs/finetune.yaml
```

## 5. Use the Checkpoint in Module 01

After training, download the best checkpoint into:

```text
training/checkpoints/
```

Then we can connect `src/ocr_engine.py` to load that checkpoint for math OCR.

## Dataset Size Advice

Start small:

- 100 labeled expression crops for a test run
- 500+ labeled crops for meaningful improvement
- Include correct, wrong, incomplete, and noisy handwriting examples

Label every image with the exact intended LaTeX.

