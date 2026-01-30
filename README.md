# Aircraft Image Classification - Deep Learning Web Application

A modern web application for aircraft image classification using multiple deep learning models including CNN, MLP, AlexNet, and ResNet50.

## Features

- **Multiple Model Support**: Compare predictions from 4 different deep learning models
- **Real-time Classification**: Upload and classify aircraft images instantly
- **Model Comparison**: Compare all models side-by-side with confidence scores
- **Prediction History**: Track your classification history
- **Modern UI**: Beautiful, responsive interface with centered layout
- **Interactive Elements**: Drag-and-drop image upload, loading indicators, and smooth animations

## Models Included

1. **CNN Model 1** - Custom Convolutional Neural Network (64x64 input)
2. **MLP Model 1** - Multi-Layer Perceptron (64x64 input)
3. **AlexNet** - Pre-trained AlexNet architecture (224x224 input)
4. **ResNet50** - Pre-trained ResNet50 architecture (224x224 input)

## Classification Classes

- Airliner
- Balloon
- Helicopter
- Dirigible
- Rockets
- Spaceshuttle
- Airplane

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd DL_APP
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure model files are in the `models/` folder:
   - `model_cnn1.pth` (97KB)
   - `model_mlp1.pth` (25MB)
   - `model_alexnet1.pth` (218MB - uses Git LFS)
   - `model_resnet1.pth` (90MB)

**Note:** Large model files (>100MB) are stored using Git LFS. When cloning, ensure Git LFS is installed and run `git lfs pull` to download the model files.

## Usage

1. Run the application:
```bash
python web_app.py
```

2. Open your browser and navigate to:
```
http://localhost:5001
```

3. Upload an aircraft image and select a model (or compare all models)

4. View the classification results with confidence scores

## Project Structure

```
DL_APP/
├── web_app.py          # Main Flask application
├── models.py           # Neural network model definitions
├── utils.py            # Utility functions (preprocessing, prediction)
├── requirements.txt    # Python dependencies
├── models/             # Trained model files (.pth)
│   ├── model_cnn1.pth
│   ├── model_mlp1.pth
│   ├── model_alexnet1.pth
│   └── model_resnet1.pth
├── templates/          # HTML templates
│   └── index.html
└── README.md           # This file
```

## Technologies Used

- **Flask** - Web framework
- **PyTorch** - Deep learning framework
- **Torchvision** - Pre-trained models and transforms
- **Bootstrap 5** - Frontend framework
- **Font Awesome** - Icons
- **Pillow** - Image processing

## Git LFS Setup

This repository uses Git LFS for large model files. If you're cloning this repository:

1. Install Git LFS:
```bash
brew install git-lfs  # macOS
# or
git lfs install
```

2. After cloning, pull the LFS files:
```bash
git lfs pull
```

## Author

**Farah Alebous**

## License

This project is for educational purposes.

