# Classical Image Segmentation Comparison

This project implements and compares three classical computer vision algorithms for segmenting floral objects from natural backgrounds. It provides a batch processing script to apply these methods to a dataset of images and visualize the results side-by-side for qualitative analysis.

## 📌 Overview

The goal of this project is to evaluate the robustness of non-deep-learning segmentation techniques on organic structures (flowers). The pipeline processes images using:
1.  **K-Means Clustering** (Color-based)
2.  **Region Growing** (Connectivity-based)
3.  **Iterative Global Thresholding** (Intensity-based)

Additionally, **Canny Edge Detection** is used to visualize structural boundaries, and **Morphological Operations** (Opening/Closing) are applied to refine the resulting masks.

## 🛠️ Requirements

The project requires Python 3.x and the following libraries:

* `numpy`
* `matplotlib`
* `opencv-python`
* `scikit-image`
* `scikit-learn`

## 🚀 Installation & Setup

1.  **Clone the repository** (or download the files).
2.  **Install dependencies** using the provided requirements file:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare Input Images:**
    * Create a folder named `input_images` in the root directory.
    * Place your `.jpg`, `.jpeg`, or `.png` images of flowers inside this folder.

## 💻 Usage

Run the batch processing script to generate segmentation results for all images in the `input_images` folder:

```bash
python segmentation_batch.py
