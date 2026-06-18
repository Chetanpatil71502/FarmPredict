# 🌾 FarmPredict — AI Crop Recommendation System

A production-ready machine learning web app that recommends the best crop to grow based on soil nutrients and climate conditions. Built with **Streamlit** and a **Random Forest Classifier**.

## ✨ Features

- **50 Indian crops** — cereals, millets, pulses, fruits, vegetables, oilseeds, spices & plantation crops
- **7 input parameters** — N, P, K (soil nutrients) + Temperature, Humidity, pH, Rainfall
- **90%+ model accuracy** on test set
- Confidence score + top-3 alternative crop suggestions
- Crop reference guide organized by category
- Clean, professional UI with no raw dataset or technical internals exposed

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌱 Supported Crop Categories

| Category | Crops |
|---|---|
| Cereals & Millets | Rice, Wheat, Maize, Bajra, Jowar, Ragi, Barley, Amaranth |
| Pulses | Chickpea, Soybean, Mungbean, Blackgram, Lentil, Peas, Groundnut, and more |
| Cash Crops | Sugarcane, Cotton, Jute |
| Oilseeds | Sunflower, Mustard, Sesame |
| Fruits | Banana, Mango, Apple, Grapes, Watermelon, Coconut, Papaya, and more |
| Vegetables | Tomato, Potato, Onion, Chilli, Brinjal, Okra, Cauliflower, Cabbage, and more |
| Spices & Plantation | Turmeric, Ginger, Cardamom, Pepper, Coffee |

## 🧠 Model Details

- **Algorithm:** Random Forest Classifier (300 estimators)
- **Features (7):** Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall
- **Training set:** 10,000 records (200 per crop, based on ICAR agronomic guidelines)
- **Test accuracy:** ~91%

## 📊 Data Sources for Production Upgrade

To upgrade to real government datasets, download from:
- [ICAR – data.gov.in](https://data.gov.in) — search "soil crop recommendation"
- [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- [FAO GAEZ India Crop Suitability](https://gaez.fao.org)

## ⚠️ Disclaimer

This tool is for educational and advisory purposes. Always validate recommendations with a local agronomist or Krishi Vigyan Kendra (KVK) before sowing.
