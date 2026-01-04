# DecodingVision 🧠👁️
Decoding Vision is an AI-powered image segmentation platform that automatically isolates the primary object in an image in real-time.

## 🔍 Problem Statement (Open Innovation)
Manual object extraction from images is time-consuming and inaccessible to non-technical users. Existing tools are either costly, inflexible, or unsuitable for real-time workflows.

## 💡 Solution
Decoding Vision leverages deep learning to provide fast, accurate, and accessible object segmentation through a simple web interface.

## 🚀 Features
- AI-based object segmentation
- Binary mask generation
- Green overlay visualization
- Real-time inference (< 3 seconds)
- Downloadable outputs
- Web-based UI

## 🛠 Tech Stack
- Python, PyTorch
- U-Net with ResNet34 encoder
- Gradio
- Hugging Face Spaces
- Google Colab (model training)

## 📊 Model Performance
- Dataset: Oxford-IIIT Pets
- IoU Score: 87.7%

## 🧩 Architecture
User → React Web → Gradio UI → PyTorch Model → Segmentation Output

## 🌐 Live Demo
👉 Hugging Face Space: [https://huggingface.co/spaces/ShreeshaS12/petvision_ai](https://huggingface.co/spaces/ShreeshaS12/petvision_ai)

## 📌 Google Technologies Used
- Google Colab – model training
- Google Drive – dataset storage

## 🔮 Future Scope
- Deployment using Google Vertex AI
- Multi-object segmentation

