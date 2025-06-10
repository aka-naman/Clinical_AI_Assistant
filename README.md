# 🩺 Clinical QA Assistant

A lightweight, GPU-accelerated question answering system designed for clinical case documents in PDF format. Built using Hugging Face Transformers and Streamlit, this project simulates the reasoning process of a junior doctor by extracting insights from uploaded clinical PDFs and answering user questions with confidence scoring and explainability.

## 🚀 Features

- 📄 Upload clinical case PDFs and extract text automatically
- 💬 Ask clinical or medical questions about the uploaded document
- 🧠 Powered by `deepset/minilm-uncased-squad2` QA model
- ⚡ GPU acceleration support via PyTorch
- 📊 Confidence scoring with floor thresholding
- 🧵 Simulated Model Context Protocol (MCP) to mimic clinical reasoning steps

## 📂 Repository Structure

- `qa_gpu_model.py`: A standalone test script that loads the QA model and runs a sample question on hardcoded context.
- `clinical_qa_app.py`: A full Streamlit web app where users can upload clinical PDFs and interact with the AI assistant via a simple UI.

## 🛠️ Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/aka-naman/clinical-qa-assistant.git
   cd clinical-qa-assistant
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run the Script**
   ```bash
   python qa_gpu_model.py
4. **Launch Streamlit app**
   ```bash
   streamlit run clinical_qa_app.py

