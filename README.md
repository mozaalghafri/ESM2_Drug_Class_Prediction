# ESM2 Drug Target Class Prediction

## Overview

This project fine-tunes Meta AI’s ESM2 protein language model to classify proteins into different drug target classes using protein sequence data retrieved from UniProt.

The model predicts whether a protein belongs to:

- Kinases
- GPCRs
- Ion Channels

---

## Features

- Protein sequence retrieval using UniProt API
- Dataset preprocessing and labeling
- Fine-tuning ESM2 transformer model
- Drug target class prediction
- Interactive inference interface using Hugging Face / Gradio

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- ESM2
- Scikit-learn
- Pandas
- Google Colab

---

## Dataset

Protein sequences were retrieved directly from the UniProt database using API queries.

---

## Project Structure

```text
ESM2_Drug_Class_Prediction/
│
├── ESM2_Drug_Target_Class_Prediction.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Expand dataset size
- Add more drug target classes
- Improve model accuracy
- Deploy a full web interface

---

## Author

Moza Alghafri
