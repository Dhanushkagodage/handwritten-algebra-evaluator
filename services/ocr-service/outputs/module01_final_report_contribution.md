# Module 01: OCR and Input Understanding

## Individual Contribution

My primary contribution to the project is the design and implementation of Module 01, the OCR and Input Understanding module for handwritten A/L algebra answers. This module is the first stage of the complete automated evaluation pipeline. Its responsibility is to accept a handwritten student answer image, extract the question and student working, identify the final answer, and convert the extracted content into a structured JSON format that can be consumed by Module 02 for mathematical reasoning and stepwise marking.

The importance of this module is that all later reasoning, marking, and feedback depend on the quality of the extracted input. If the handwritten answer is not correctly read or if the steps are not preserved in order, the reasoning module may evaluate the wrong expression or miss important student working. Therefore, my work focused not only on OCR, but also on preprocessing, step segmentation, OCR correction, output structuring, and API integration.

## Role of Module 01 in the Overall System

The overall system follows a three-module architecture:

1. Module 01: OCR and Input Understanding
2. Module 02: Multi-Agent Mathematical Reasoning and Stepwise Marking
3. Module 03: Stepwise Feedback Generation

Module 01 acts as the extraction layer. It receives the raw handwritten answer image from the user and converts visual information into machine-readable structured content. The module does not mark the answer and does not generate feedback. Its output is designed specifically as the input contract for Module 02.

The final output shape produced by Module 01 is:

```json
{
  "reasoning_input": {
    "question_text": "Resolve into partial fractions: (2x + 3)/(x - 1)^2",
    "student_steps": [
      {
        "step_id": 1,
        "content": "(2x + 3)/(x - 1)^2 = A/(x - 1) + B/(x - 1)^2"
      },
      {
        "step_id": 2,
        "content": "2x + 3 = A(x - 1) + B"
      }
    ],
    "final_answer": "2/(x - 1) + 5/(x - 1)^2"
  }
}
```

This structure keeps the module boundary clean. Module 02 receives only the question text, ordered student steps, and final answer, without needing to handle image files, OCR confidence values, bounding boxes, or OCR engine details.

## Problem Addressed by Module 01

The main problem addressed by this module is converting real handwritten algebra answer images into structured text that can be understood by an automated reasoning system. This is challenging because handwritten algebra answers contain mixed content: natural language phrases, mathematical symbols, fractions, superscripts, variables, crossed-out work, and step-by-step working. In real answer photos, additional difficulties occur due to skewed camera angles, notebook lines, shadows, background objects, low contrast, and irregular handwriting styles.

Unlike printed OCR or isolated symbol recognition, this module must understand answer sheets as complete solution attempts. It must preserve the order of working steps, separate intermediate steps from the final answer, and output the result in a stable JSON schema. This makes the module an input-understanding component rather than a simple OCR wrapper.

## Objectives of Module 01

The objectives of Module 01 are:

1. Accept a handwritten algebra answer image through an API endpoint.
2. Store the uploaded image safely for processing and traceability.
3. Preprocess noisy handwritten images to improve text visibility.
4. Segment the answer image into step or line regions for local OCR experiments.
5. Apply OCR using both local OCR baselines and a high-accuracy vision model backend.
6. Preserve the order of the student's algebraic working.
7. Extract or accept the question text when available.
8. Identify the final answer from the student's visible work.
9. Return a clean JSON output suitable for Module 02.
10. Save successful results for later evaluation and reporting.

## Implemented System Architecture

The implemented Module 01 system is a FastAPI-based service. The main endpoint is:

```text
POST /extract
```

The user uploads a handwritten answer image. The API returns structured JSON in the `reasoning_input` format. The service supports two OCR modes:

1. `openai_vision`: the recommended high-accuracy mode using a vision-capable language model.
2. `local`: a research baseline using preprocessing, segmentation, EasyOCR, and optional pix2tex.

The implemented files are:

| File | Purpose |
|---|---|
| `app.py` | FastAPI application and `/extract` endpoint |
| `src/openai_vision_ocr.py` | High-accuracy vision OCR backend and JSON parser |
| `src/preprocess.py` | OpenCV preprocessing for raw answer images |
| `src/segment.py` | Step/line segmentation for local OCR experiments |
| `src/ocr_engine.py` | EasyOCR, pix2tex, hybrid, and manual OCR support |
| `src/correction.py` | Algebra-oriented OCR correction rules |
| `src/structure_output.py` | Conversion of OCR results into Module 02 JSON |
| `src/result_store.py` | Saves extraction outputs for analysis |

## Processing Pipeline

The implemented pipeline is:

```text
Uploaded image
    -> Image saved to data/raw/
    -> OCR mode selected
    -> High-accuracy vision extraction or local OCR baseline
    -> JSON validation and cleanup
    -> reasoning_input output
    -> Result saved to outputs/
```

For the local OCR baseline, the pipeline is:

```text
Uploaded image
    -> OpenCV preprocessing
    -> Step/line segmentation
    -> EasyOCR text extraction
    -> Optional pix2tex math OCR
    -> Algebra correction rules
    -> Structured JSON output
```

For the recommended working system, the pipeline is:

```text
Uploaded image
    -> Vision model receives image and extraction prompt
    -> Model transcribes question, steps, and final answer
    -> Response is validated as JSON
    -> Clean reasoning_input is returned
```

## Preprocessing Contribution

I implemented an OpenCV preprocessing pipeline to improve raw answer images before OCR. The preprocessing stage includes:

1. Reading the uploaded answer image.
2. Detecting and cropping unnecessary dark background areas.
3. Attempting to isolate the paper region from the full camera image.
4. Converting the image to grayscale.
5. Normalizing illumination to reduce shadow effects.
6. Applying thresholding to improve contrast between handwriting and paper.
7. Applying skew correction where possible.
8. Saving the preprocessed image to `data/preprocessed/`.

This part is important because real student answer images are not clean scanned documents. They may include laptop backgrounds, shadows, tilted pages, and ruled notebook lines. Preprocessing improves the probability that OCR models receive a cleaner input image.

## Segmentation Contribution

I implemented a segmentation component that attempts to crop handwritten answer lines or working steps from the preprocessed image. The segmentation code uses foreground masks, line removal, contour detection, projection profiles, box merging, and filtering rules to identify possible handwritten step regions. Each detected region is saved as a separate image in `data/regions/`.

The purpose of segmentation is to support step-level OCR and later step-level reasoning. Instead of treating the whole page as one block of text, the system attempts to preserve the student's solution sequence. This is important because algebra marking is stepwise: each intermediate line may show correct reasoning, a mistake, or a missing transformation.

## OCR Backend Contribution

During implementation, I tested local OCR approaches such as EasyOCR and pix2tex. EasyOCR was useful as a general text OCR baseline, while pix2tex was considered for mathematical expression recognition. However, experiments on real handwritten answer photos showed that local OCR struggled with noisy images, fractions, superscripts, notebook lines, and informal handwriting. Example errors included incorrect recognition of variables, missing equality signs, and meaningless symbol sequences.

To make the system fully usable for the current project stage, I implemented a high-accuracy vision OCR backend. This backend sends the uploaded image to a vision-capable language model with a strict extraction prompt. The model is instructed to:

1. Extract what the student wrote.
2. Preserve the order of working steps.
3. Avoid marking or correcting the answer.
4. Put the last clear result into `final_answer`.
5. Return only valid JSON in the required Module 02 format.

This backend gave much better results on realistic handwritten answer photos than the local OCR baseline. Therefore, the final working system uses the vision backend as the recommended mode, while keeping the local OCR pipeline as a research baseline for comparison and future improvement.

## API Contribution

I implemented a FastAPI service so that Module 01 can be used independently by other modules. The `/extract` endpoint allows the user to upload an image directly. The endpoint does not require the user to manually type steps. It returns only the structured JSON needed by Module 02.

The endpoint supports:

1. Image upload using multipart form data.
2. Optional typed question text if the question is not clearly visible in the image.
3. OCR mode selection between `openai_vision` and `local`.
4. JSON response validation.
5. Error handling for missing API keys, invalid OCR modes, and processing failures.
6. Saving successful outputs to `outputs/results.json` and per-image result files.

This API design makes the module reusable. Module 02 can call the API and receive a clean JSON object without knowing how OCR is performed internally.

## Dataset and Experiment Contribution

I collected a small internal dataset of handwritten algebra answer images with different answer conditions, including correct answers, wrong answers, missing steps, and varied handwriting. These images were stored in `data/raw/` and used to test preprocessing, segmentation, and OCR extraction.

Because the available dataset is small, the current system does not train a new OCR model from scratch. Instead, the dataset is used for:

1. Testing OCR behavior on realistic answer images.
2. Comparing local OCR with the vision backend.
3. Identifying common OCR failure cases.
4. Preparing cropped region images for future labeling.
5. Creating a foundation for future pix2tex fine-tuning.

I also prepared training support scripts to create labeling datasets and export labeled step images for future math OCR fine-tuning. This supports the future research direction of improving a local math OCR model using the collected handwritten algebra samples.

## Research Contribution

The research contribution of Module 01 is not the creation of a new OCR foundation model. Instead, the contribution is the design and implementation of an OCR input-understanding pipeline for handwritten algebra evaluation. The contribution includes:

1. A working API-based OCR and input-understanding service for handwritten algebra answers.
2. A structured JSON contract that connects handwritten image input to downstream mathematical reasoning.
3. A preprocessing and segmentation pipeline for realistic answer photos.
4. A comparison pathway between local OCR models and high-accuracy vision-language OCR.
5. A practical solution for extracting ordered algebra steps from real handwritten answer images.
6. A dataset preparation workflow for future fine-tuning of math OCR models.

This contribution is suitable for the final-year research project because it addresses a specific research gap: existing OCR systems often handle either general handwritten text or isolated mathematical expressions, but this project requires mixed-content exam answer understanding with step ordering and module-to-module structured output.

## Evaluation Plan

The module can be evaluated using the collected handwritten answer images. For each image, a manually prepared ground-truth JSON file can be compared with the system output.

The following evaluation metrics are suitable:

1. Question extraction accuracy: whether the visible question is correctly extracted.
2. Step count accuracy: whether the correct number of student steps is identified.
3. Step order accuracy: whether the extracted steps preserve the original working order.
4. Step content similarity: similarity between extracted step text and manually labeled ground truth.
5. Final answer accuracy: whether the final answer is correctly extracted.
6. JSON validity rate: percentage of responses that match the required schema.
7. Backend comparison: local OCR output quality compared with the vision backend.
8. Latency: average processing time per image.

For reporting, the evaluation can compare:

| Backend | Strength | Weakness |
|---|---|---|
| EasyOCR | Fast and local | Poor recognition of algebra handwriting |
| EasyOCR + pix2tex | Better math-focused direction | Slow and unreliable without fine-tuning |
| Vision backend | High extraction quality on real images | Requires API access and has cost/privacy considerations |

## Limitations

The current module has several limitations:

1. Local OCR accuracy is low on noisy handwritten algebra images.
2. The high-accuracy backend depends on an external vision model API.
3. Very unclear handwriting may still produce incorrect extraction.
4. The system extracts student work but does not verify mathematical correctness.
5. The collected dataset is currently small and should be expanded for stronger evaluation.
6. Fine-tuned local math OCR has not yet been completed.
7. Cloud-based OCR introduces cost, latency, and privacy considerations.

These limitations are acceptable at the current stage because the module has a fully working extraction service and a clear research path for improving local OCR performance.

## Future Work

Future improvements for Module 01 include:

1. Expanding the handwritten algebra dataset with more answer styles and question types.
2. Creating manually labeled ground-truth JSON files for all collected answer images.
3. Fine-tuning pix2tex or another math OCR model using cropped handwritten algebra regions.
4. Improving automatic page boundary detection and perspective correction.
5. Adding confidence scoring for extracted steps.
6. Adding a teacher review interface for correcting OCR outputs and building training data.
7. Comparing local OCR, fine-tuned math OCR, and vision model OCR using formal metrics.
8. Improving privacy by supporting a stronger local OCR option in future versions.

## Final Report Summary Paragraph

In summary, my contribution focused on implementing Module 01: OCR and Input Understanding. I developed a FastAPI-based service that accepts handwritten algebra answer images and returns a structured `reasoning_input` JSON object containing the question text, ordered student working steps, and final answer. I implemented preprocessing, segmentation, OCR engine integration, correction, JSON structuring, result saving, and API delivery. Initial experiments showed that traditional local OCR tools such as EasyOCR and pix2tex struggled with realistic handwritten algebra photos. Therefore, I integrated a high-accuracy vision OCR backend as the recommended working mode while preserving the local OCR pipeline as a baseline for research comparison and future fine-tuning. This module provides the essential bridge between raw handwritten student answers and the downstream reasoning and feedback modules.
