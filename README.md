# 🖼️ Projet CLIP : Contrastive Language-Image Pre-training

**Réalisé par :** Belguith Ahmed Rami & Benboubaker Majdi  
**Date :** Janvier 2026  
**Cadre :** Master 2 - Université de Montpellier  

---

## 📖 Introduction
Ce projet implémente un modèle **CLIP** (Contrastive Language-Image Pre-training) développé avec TensorFlow/Keras. Son objectif est de créer un espace latent commun permettant d'aligner sémantiquement des représentations visuelles et textuelles. 

Le système permet de réaliser deux types de recherches multimodales :
*   **Texte ➔ Images :** Trouver les images correspondant à une requête textuelle.
*   **Image ➔ Textes :** Trouver les descriptions (captions) les plus pertinentes pour une image donnée.

## 📊 Dataset
*   **Source :** Sous-ensemble du dataset Flickr.
*   **Volume :** 800 paires (image, texte).
*   **Classes :** 4 classes équilibrées (ball, bike, dog, water), soit 200 images par classe.
*   **Format :** Images redimensionnées en `224x224` pixels.
*   **Répartition :** 80% Entraînement / 10% Validation / 10% Test.

---

## 🏗️ Architecture du Pipeline

Le projet est construit en trois phases principales :

### 1. Classifieur d'Images (CNN)
*   **Architecture :** 3 blocs `Conv2D` (32, 64, 128 filtres) + `MaxPooling2D` ➔ `Flatten` ➔ `Dense(256)` ➔ `Dropout(0.2)` ➔ Sortie `Dense(4)`.
*   **Performances :** Validation Accuracy ~38.12%. L'objectif principal de ce module est d'agir comme un extracteur de caractéristiques (feature extractor) basique.

### 2. Classifieur de Textes (SmallBERT)
*   **Architecture :** `Positional Embedding` (vocabulaire de 10 000 mots, longueur de séquence de 32) + 2 `Transformer Blocks` (4 têtes d'attention, `embed_dim=128`, `ff_dim=256`).
*   **Pooling :** Utilisation d'un `MaskedMean` pooling personnalisé (ignorant les tokens de padding) à la place du token classique `[CLS]`.
*   **Performances :** Validation Accuracy ~80.00%.

### 3. Modèle CLIP
*   **Encodeurs :** Dérivés des classifieurs ci-dessus en remplaçant les têtes de classification par une couche `Dense(128)` suivie d'une **Normalisation L2**.
*   **Perte (Loss) :** Implémentation d'une perte contrastive (`ClipLossLayer` avec `temperature=0.07`) visant à rapprocher les paires correctes sur la diagonale de la matrice de similarité.

---

## 🔍 Inférence et Recherche
L'inférence repose sur le calcul des similarités cosinus (obtenu par un simple produit matriciel grâce à la normalisation L2) entre la requête et un index d'embeddings pré-calculés.
1.  **Prétraitement :** Redimensionnement de l'image (224x224) ou Tokenization du texte (Max len 32).
2.  **Encodage :** Passage dans l'encodeur CLIP respectif pour obtenir un vecteur `1x128` normalisé.
3.  **Similarité :** Produit matriciel avec l'index pré-calculé (images ou textes).
4.  **Top-K :** Renvoi des $K$ (ex: 5) meilleurs résultats triés par leur score de similarité.

---

## ⚙️ Configuration Technique
*   **Langage :** Python 3.10 / 3.12
*   **Frameworks :** TensorFlow 2.15.0 / Keras, Pandas, NumPy, Matplotlib
*   **Matériel :** Entraîné sur GPU (Google Colab Pro Tesla T4 / PC MSI GeForce GTX 1650 Ti, CUDA 12.2 / 12.9)
