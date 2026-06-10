# DCDSM

> **DCDSM: An effective method for driver synonymous mutation prediction with a dual-channel fusion network**

DCDSM is a deep learning model designed for **driver synonymous mutation prediction**.  
In this study, we hypothesize that if a synonymous mutation has been validated as pathogenic in the germline and also appears as a somatic mutation in tumors, then it may be a cancer driver synonymous mutation.

To enhance the representation ability of DNA sequences, DCDSM integrates features from three large-scale DNA sequence pre-trained language models:

- **DNABERT**
- **HyenaDNA**
- **ChemicalBERT**

In addition, the model incorporates eight types of handcrafted features and processes them using a **dual-channel fusion network**. Based on this design, DCDSM applies **CNN-BiLSTM** to extract both local and global features, and then uses a **self-attention mechanism** to perform feature fusion, improving the prediction performance of driver synonymous mutations.

---

## 📁 Project Structure

```text
DCDSM/
├── data/
│   └── feature/
├── environment/
│   └── environment.yml
├── model/
│   ├── get_DNABERT_feature.py
│   ├── get_chemical_feature.py
│   ├── get_HyenaDNA_feature.py
│   ├── lightGBM.py
│   ├── DCDSM.py
│   └── best_model_fold_9.h5
└── pretrained_model/
```

---

## 📊 Data

The `data/feature/` directory contains the feature data constructed and used in this study, including:

- Training and test set features constructed based on **HGMD**, **ClinVar**, and **gnomAD**;
- Features of the second test set constructed based on **SpliceVar**;
- Features used for case analysis;
- `data/feature/LightGBM/`, which stores the feature selection results.

---

## 🧠 Model

The `model/` directory contains the core code and model files of the project, including:

| File | Description |
|---|---|
| `get_DNABERT_feature.py` | Extracts DNABERT features |
| `get_chemical_feature.py` | Extracts ChemicalBERT features |
| `get_HyenaDNA_feature.py` | Extracts HyenaDNA features |
| `lightGBM.py` | Performs feature selection based on LightGBM |
| `DCDSM.py` | Main implementation of the DCDSM model |
| `best_model_fold_9.h5` | Saved best-performing model |

---

## 🔬 Pre-trained Models

The `pretrained_model/` directory contains the weights of the pre-trained genomic language models required by this project.

---

## ⚙️ Environment Setup

We recommend using **Anaconda** to create a Python virtual environment.  
Please refer to the following file for the complete environment configuration:

```text
environment/environment.yml
```

The main dependencies are listed below:

```text
python=3.6.13
tensorflow=2.6.0
scikit-learn=0.24
```

Create the conda environment for DCDSM:

```bash
conda env create -f environment/environment.yml
```

Activate the environment:

```bash
conda activate DCDSM
```

> If the environment name defined in `environment.yml` is not `DCDSM`, please replace `DCDSM` in the command above with the corresponding environment name.

---

## 🚀 Usage

Please refer to the template data in the `/data` directory.  
This directory contains various feature data and synonymous mutations in VCF format.

If you want to run DCDSM on your own data, please first process your data into the same format as the template data.

View the help documentation:

```bash
python model/DCDSM.py -h
```

Run the model:

```bash
python model/DCDSM.py
```

---

## 📝 Notes

- Make sure that the required pre-trained model weights are placed in the `pretrained_model/` directory.
- Make sure that the input data format is consistent with the template data provided in the `data/` directory.
- If you use custom data, it is recommended to perform feature extraction and feature selection before running the model.

---

## Citation

If you use DCDSM in your research, please cite the corresponding paper or repository.
