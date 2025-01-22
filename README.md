# Voice Emotion Recognition

Voice Emotion Recognition is a machine learning project designed to classify and recognize emotions based on audio signals. It leverages advanced signal processing techniques and machine learning algorithms to predict the emotional state of a speaker from their voice.

---

## Features

- **Emotion Detection**: Recognizes emotions such as happiness, sadness, anger, fear, and more.
- **Audio Preprocessing**: Includes noise reduction, normalization, and feature extraction (MFCC, spectrograms, etc.).
- **Model Training**: Utilizes machine learning/deep learning models such as Support Vector Machines (SVM), Random Forests, or Neural Networks.
- **Real-time Prediction**: Option to predict emotions from live audio input.

---

## Installation

### Prerequisites

Ensure you have Python 3.7 or later installed on your system. Additionally, install the following dependencies:

```bash
pip install numpy pandas scikit-learn librosa tensorflow keras matplotlib
```

### Clone the Repository

```bash
https://github.com/GKRBROS/Voice_Emotion_Recognition.git
cd voice-emotion-recognition
```

---

## Usage

### Data Preparation

1. **Dataset**: Obtain an emotional speech dataset such as:
   - [RAVDESS](https://zenodo.org/record/1188976)
   - [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
2. Place the dataset in the `data/` directory.

### Training the Model

Run the following command to preprocess the data and train the model:

```bash
python train.py
```

### Real-time Emotion Detection

Use the following command to test real-time emotion detection:

```bash
python predict.py
```

---

## Project Structure

```
voice-emotion-recognition/
├── data/              # Directory for datasets
├── models/            # Trained models
├── utils/             # Helper scripts for preprocessing and evaluation
├── train.py           # Script for training the model
├── predict.py         # Script for predictions
├── README.md          # Project documentation
```

---

## Technologies Used

- **Python**: Programming language
- **Librosa**: Audio processing library
- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Machine learning library

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your branch.
4. Submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- Datasets used for training and testing.
- Open-source tools and libraries that made this project possible.

---

## Contact

For questions or suggestions, feel free to reach out:

- **Name**: Gokul Kiran Radhakrishnan
- **Email**: gokulkiran@example.com
- **LinkedIn**: [linkedin.com/in/gokul-kiran](https://linkedin.com/in/gokul-kiran)

---

Happy coding!
