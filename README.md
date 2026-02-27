# Sign Language Translator

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-MediaPipe-green)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

A real-time **Sign Language to Text translator** implemented as a Python application using computer vision and an optimized machine learning model.

The project allows you to create your own dataset from recorded gesture images, extract landmarks, generate angle-based features, train a custom model, and perform real-time gesture recognition.

---

## Key Features

-  Fast and efficient recognition of **static and dynamic gestures**
-  Minimalistic interface with:
  - Predefined gesture text field
  - Current detected gesture display
  - Confidence score bar
- Ability to edit text fields using gestures
-  Model export in `.pkl` format for optimization and sharing
-  Flexible dataset recording and editing module
-  Automatic landmark extraction
- Use of **landmark angles instead of raw coordinates**  
  (removes influence of camera distance and hand position)
- Stable performance even with small datasets (~100 images per gesture)

---

## How It Works

```
Camera -> MediaPipe Hand Detection -> Landmark Extraction -> Angle Calculation -> Feature Vector (.csv) -> Machine Learning Model -> Prediction -> Text Output
```

### Angle Calculation Formula

To eliminate spatial dependency, angles between three landmarks are calculated:

$\theta = \arccos\left(\frac{(A - B) \cdot (C - B)}{\|A - B\| \cdot \|C - B\|}\right)$

Where: 

-**B**  the vertex of the angle,
 
-**A, B, C** - 3D landmark points detected by MediaPipe,

-**$\theta$** - resulting angle between the two vectors.

---

## Installation

```bash
git clone https://github.com/Felix754/sign-language-translator.git
cd sign-language-translator

# Recommended: create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r REQUIREMENTS.txt
```



## Project Structure

```
sign-language-translator/
│
├── REQUIREMENTS.txt       # Requiriments
├── asl_model.pkl          # Trained model for testing
├── asl_angles.csv         # Extracted landmarks
├── main.py                # Real-time gesture recognition app
├── model.py               # Model training module
├── landmarks.py           # Landmark extraction module
├── dataset.py             # Dataset creation module
└── data/                  # Recorded gesture images
```

---

## Project Modules

### `main.py`
Main application for real-time gesture recognition.  
Requires a trained `.pkl` model file.

### `model.py`
Responsible for training the machine learning model.  

**Input:** `.csv` file with extracted gesture features  
**Output:** `.pkl` trained model file  



### `landmarks.py`
Extracts hand landmarks from gesture images inside `/data`.  

**Output:** `.csv` file containing angle-based feature vectors.



### `dataset.py`
Creates a dataset from captured gesture images.

Before running, specify arguments (use `-h` to see available options):

```bash
python dataset.py -h
```

To create one or multiple gestures, specify their names inside the `/gestures/` array.

Manual renaming of gesture folders is not recommended.

---

## Dependencies

All dependencies are listed in `REQUIREMENTS.txt`.

Main libraries:

- NumPy  
- Scikit-learn  
- OpenCV  
- Pandas  
- Argparse  
- MediaPipe *(older version, since `hands` is deprecated in newer releases)*  
- Pickle  

---

## Future improvements/ideas 

- Improved dynamic gesture detection
- AI-powered translation system
- Web interface or mobile application integration
- Cross-platform deployment

---

## License

This project is open-source.  
You can use the **MIT License** (recommended) or choose another license depending on your needs.
