# Creature Image Classification

### Project Summary:

This project creates a machine learning model to classify multiple creatures' images into 4 categories: cow, elephant, horse, spider, using Tensorflow and Keras.

This notebook: `Creature_Image_Classification.ipynb` was used to train a creature classification challenge held in Kaggle conducted by OpenCV University.

---

### Dataset Description:

The dataset consists of `3997` images for **4 animal classes**. Here are some sample images from these four classes:

[TODO add image here]

The original dataset can be accessed on the [Kaggle Dataset: Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) 
---

### Machine Learning Model Architecture:

This model is composed of a Neural Network from Scratch utilizing CNN architecture. 
First, added 4 `Feature Extraction Layers (Conv2d -> BatchNorm -> ReLU -> MaxPooling -> Dropout)`.
Then, used `Global Average Pooling 2D Layer`.
Lastly, added `Dense Layer` and `Output Layer`.

---

### Data Augmentation:

For the training dataset, I applied the following data augmentation to avoid overfitting:

```
data_augmentation_pipeline = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),    # Random horizontal flip
    tf.keras.layers.RandomRotation(0.1),         # Random rotation (up to 10 degrees)
])
```

---

### Training Hyperparameters:

* Epochs: ` 111`
  
* Optimizer: `Adam`

* Initial Learning Rate: `0.001`

* Batch size: `32`

---

### Loss and Accuracy:

![](./visuals/creature_classification_loss_accuracy.png?raw=true)

---

### Inferences:

![](./visuals/creature_classification_inference-transformed.png?raw=true)

---

### Confusion Matrix:

![](./visuals/creature_classification_confusion_matrix.png?raw=true)

---

### Accuracy on Test Dataset for Kaggle Submission

The configurations discussed above, yielded a score of **0.84859** on the Kaggle's Leaderboard.

![](./visuals/creature_classification_kaggle_leaderboard.png?raw=true)
