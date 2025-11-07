# 🔥 Fitbit Calories Burned Predictor

A web-based **machine learning app** built with **Streamlit** and **PyTorch** that predicts **daily calories burned** based on Fitbit activity data.

This app uses a **custom-trained MLP (Multi-Layer Perceptron)** neural network model saved as `best_model.pth`, along with a fitted `StandardScaler` (`scaler.save`), to make predictions on 12 Fitbit activity metrics.

---

## 🚀 Features

- 🧠 **Custom PyTorch Model** — Predict calories burned from Fitbit data.
- ⚙️ **Input Interface** — User-friendly Streamlit UI for entering daily activity data.
- 📈 **Dynamic Predictions** — Displays real-time calorie burn results and insights.
- 💡 **Custom Styling** — Modern layout with interactive feedback and styled components.
- 💾 **Cached Model Loading** — Efficient performance using Streamlit’s caching features.

---

## 🧩 Tech Stack

- [Streamlit](https://streamlit.io/) — Web app framework
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Pandas](https://pandas.pydata.org/) — Data manipulation
- [Scikit-learn](https://scikit-learn.org/) — Data scaling (`StandardScaler`)
- [Joblib](https://joblib.readthedocs.io/) — Model persistence
- [NumPy](https://numpy.org/) — Numerical operations

---

## 📁 Project Structure

