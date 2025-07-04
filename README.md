# Alzheimer's Disease Classification using CNNs

This project applies convolutional neural networks (CNNs) to classify Alzheimer's Disease stages from MRI scans. Early and accurate detection of Alzheimer's is critical for timely treatment, and this project explores deep learning techniques to support diagnostic automation. This project demonstrates how deep learning can be applied to real-world medical imaging data to support clinical decision-making.

## Models Implemented

1. **Baseline CNN Model** — Basic architecture trained from scratch
2. **Augmented CNN Model** — Utilized `ImageDataGenerator` for real-time augmentation
3. **Transfer Learning Model** — Used VGG16 with frozen base layers
4. **Fine-Tuned Model** — Unfroze top layers of VGG16 for additional training
5. **Hyperparameter-Tuned Model** — Optimized with `kerastuner.RandomSearch`

## Tools & Libraries

- **Data**: Hugging Face Datasets (`datasets.load_dataset`), PIL (image handling)
- **Deep Learning**: TensorFlow, Keras, Keras Tuner
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: Scikit-learn (classification report, confusion matrix)
- **Other**: NumPy, Pandas, class balancing with `sklearn.utils.class_weight`

## Results

Each model was evaluated on validation and test sets with metrics including:
- Accuracy
- Confusion matrix
- Precision, recall, and F1-score
