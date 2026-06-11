# 🔬 Advanced Ecological Modelling: GAN-Enriched Data Enhancement

## 📌 Project Overview
Predictive models frequently suffer from severe class imbalances, data scarcity, and fragmented tracking metrics. This repository presents an advanced machine learning solution: a Tabular Generative Adversarial Network (GAN) framework designed to target minority classes, synthesize high-fidelity feature distributions, and optimize downstream classification accuracy.

> 💡 ANALYTICAL NOTE:
> While originally developed on ecological modeling data (Code_sk), this decoupled pipeline architecture directly translates to live-service technology ecosystems requiring complex user behavior simulation, data enrichment, or synthetic telemetry generation (e.g., gaming analytics, automated balance testing, and player matchmaking profiles).

---

## 🏗️ Architecture & Decoupled Engine
Instead of utilizing standard experimental scratchpad notebooks, this framework is engineered using a scalable, production-grade dual-module structure:

* **models.py**: Houses the independent network architectures. Implements a dense Generator network utilizing Batch Normalization and LeakyReLU layers to synthesize mock metrics, and a Discriminator network optimized via binary cross-entropy to evaluate data authenticity.
* **pipeline.py**: The execution engine. Automates raw data ingestion, handles minority oversampling via class-specific GAN loops, executes Recursive Feature Elimination (RFECV) to strip out analytical noise, and trains an optimized Random Forest Classifier.

---

## 🛠️ Technology Stack
* Core Language: Python
* Deep Learning Framework: TensorFlow / Keras
* Machine Learning & Feature Selection: Scikit-learn (RFECV, RandomForestClassifier)
* Data Engineering & Visualization: Pandas, NumPy, Seaborn, Matplotlib

---

## 📂 Repository Structure

* **GAN-Enriched-Data-Enhancement/** (Root)
    * **data/** *(Folder - Securely retained locally via .gitignore)*
        * `.gitkeep` (Maintained folder structure placeholder)
    * `models.py` (Neural network adversarial architectures)
    * `pipeline.py` (End-to-end oversampling and classification pipeline)
    * `requirements.txt` (Production dependencies and library constraints)

> 🔒 DATA PRIVACY NOTE:
> In accordance with academic data privacy protocols, the raw underlying research datasets are actively excluded via .gitignore and retained locally. The pipeline automatically falls back to generating structured simulation vectors if executed without local storage tracking.

---

## 🚀 Key Performance Indicators & Features

* **GAN-Driven Class Balancing**: Automatically evaluates structural class imbalances and dynamically spins up dedicated generative pipelines to normalize minority distributions up to a targeted threshold.
* **Recursive Feature Elimination (RFECV)**: Filters out feature noise using a 10-Fold Stratified Cross-Validation mask, optimizing computational performance by keeping only the most predictive metrics.
* **Production Invariants**: Written with strict type hinting, robust system path checking via os.path, and clean separation of concerns.
