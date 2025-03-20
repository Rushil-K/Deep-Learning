🔢 Image Recognition 0 to 9
A Streamlit-based Image Recognition App that predicts handwritten digits (0-9) using a deep learning model trained on the MNIST dataset. This app allows users to upload multiple scanned handwritten digit images and provides accurate predictions along with probability distributions.

🌐 Live Demo
👉 Try the App Here

📌 Features
✔️ Predicts digits (0-9) from scanned handwritten images
✔️ Accepts multiple image uploads at once
✔️ Preprocessing handles background detection & correction
✔️ Displays probability scores for better understanding
✔️ Interactive bar chart visualization of predictions
✔️ Optimized for clarity and ease of use

🛠️ Tech Stack
Python 3
Streamlit (for the UI)
TensorFlow / Keras (for the deep learning model)
NumPy & PIL (for image processing)
gdown (for downloading the model from Google Drive)
📂 Setup & Installation
Follow these steps to set up the project locally:

1️⃣ Clone the Repository
sh
Copy
Edit
git clone https://github.com/Rushil-K/Image-Recognition-0-to-9.git
cd Image-Recognition-0-to-9
2️⃣ Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Streamlit App
sh
Copy
Edit
streamlit run app.py
📸 Usage Guide
🖼️ Uploading Images
Scan and upload your handwritten digit images.
Ensure high visibility (clear writing, minimal noise).
Upload JPG, PNG, or JPEG format files.
📊 Interpreting the Output
The app will display the predicted digit.
A bar chart visualization will show the probability scores.
🏗️ Project Contributors
Rushil Kohli
Navneet Mittal
🔗 Follow Rushil Kohli on GitHub 👉

📜 License
This project is licensed under the Apache 2.0 License.
📄 See the full license here: LICENSE

🎯 Future Improvements
✅ Support for handwritten text (not just digits)
✅ Enhancement in noise reduction and preprocessing
✅ Deployment on cloud platforms
🚀 Star this repository if you find it useful! ⭐
