# X-Ray Image Classification

This project focuses on classifying chest X-ray images to detect pneumonia. The project uses convolutional neural networks (CNNs) implemented with PyTorch.

## Project Structure
001_dataset_download.ipynb 002_image_classification_norm.ipynb 002_image_classification_scale.ipynb 002_image_classification.ipynb 003_image_normalization.ipynb env/ .gitignore CACHEDIR.TAG Lib/ site-packages/ pyvenv.cfg Scripts/ pycache/ ... xray_classifier.pth


## Notebooks

- **001_dataset_download.ipynb**: Downloads and extracts the dataset from Kaggle.
- **002_image_classification.ipynb**: Implements the image classification model.
- **002_image_classification_norm.ipynb**: Implements image normalization and classification.
- **002_image_classification_scale.ipynb**: Implements image scaling and classification.
- **003_image_normalization.ipynb**: Additional image normalization techniques.

## Setup

1. Clone the repository.
2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Download Dataset**: Run the [001_dataset_download.ipynb](http://_vscodecontentref_/6) notebook to download and extract the dataset.
2. **Train Model**: Run the [002_image_classification.ipynb](http://_vscodecontentref_/7) notebook to train the model.
3. **Normalize and Train**: Run the [002_image_classification_norm.ipynb](http://_vscodecontentref_/8) notebook to normalize images and train the model.
4. **Scale and Train**: Run the [002_image_classification_scale.ipynb](http://_vscodecontentref_/9) notebook to scale images and train the model.

## Model

The trained model is saved as [xray_classifier.pth](http://_vscodecontentref_/10). You can load this model for inference or further training.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) by Paul Mooney on Kaggle.

## Contact

For any questions or suggestions, please open an issue or contact the project maintainers.