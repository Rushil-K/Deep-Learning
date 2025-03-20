# Image Recognition 0 to 9  

A **Streamlit-based Image Recognition App** that predicts handwritten digits (0-9) using a deep learning model trained on the **MNIST dataset**.  
This app allows users to **upload multiple scanned handwritten digit images** and provides accurate predictions along with probability distributions.  

---

## Live Demo  
[Try the App Here](https://deep-learning-daqsskxacd8e5j7hketto2.streamlit.app/)  

---

## Features  
- Predicts digits (0-9) from scanned handwritten images  
- Accepts multiple image uploads at once  
- Preprocessing handles background detection and correction  
- Displays probability scores for better understanding  
- Interactive bar chart visualization of predictions  
- Optimized for clarity and ease of use  

---

## Tech Stack  
- Python 3  
- Streamlit (for the UI)  
- TensorFlow / Keras (for the deep learning model)  
- NumPy & PIL (for image processing)  
- gdown (for downloading the model from Google Drive)  
- Figma (for slicing images in the dataset)  

---

## Setup & Installation  

Follow these steps to set up the project locally:  

### Clone the Repository  
```sh
git clone https://github.com/Rushil-K/Image-Recognition-0-to-9.git
cd Image-Recognition-0-to-9

---

## Usage Guide  
The app currently supports only **black and white images**, including both **white text on a black background** and **black text on a white background**.  

### Uploading Images  
- **Scan and upload** your handwritten digit images or images from the web.  
- Ensure **high visibility** with clear writing and minimal noise.  
- Upload files in **JPG, PNG, or JPEG** format.  

### Image Processing  
- The dataset uploaded to **Google Drive** in a **ZIP folder** contains images used for training.  
- **Figma** was used to **slice** these images to prepare a structured dataset.  

### Interpreting the Output  
- The app will **display the predicted digit**.  
- A **bar chart visualization** will show the probability scores.  

---

## Model Performance  
The deep learning model was trained on the MNIST dataset with the following results:  

| Epoch | Accuracy | Loss | Validation Accuracy | Validation Loss |
|--------|-----------|-------|------------------|----------------|
| 1 | 82.54% | 0.6226 | 89.25% | 0.3282 |
| 2 | 95.85% | 0.1386 | 98.09% | 0.0631 |
| 3 | 96.90% | 0.1056 | 98.81% | 0.0388 |
| 4 | 97.31% | 0.0929 | 98.17% | 0.0644 |
| 5 | 97.47% | 0.0853 | 98.90% | 0.0350 |
| 6 | 97.73% | 0.0785 | 98.68% | 0.0494 |
| 7 | 97.97% | 0.0683 | 98.78% | 0.0423 |
| 8 | 97.88% | 0.0751 | 99.32% | 0.0229 |
| 9 | 98.18% | 0.0653 | 98.99% | 0.0332 |

- The model successfully predicted **87% of printed font numbers**.  
- The model correctly identified **58% of handwritten numbers**, indicating scope for improvement in handwritten digit recognition.  

---

## Project Contributors  
- **Rushil Kohli**  
- **Navneet Mittal**  

Follow Rushil Kohli on GitHub: [GitHub Profile](https://github.com/Rushil-K)  

---

## License  
This project is licensed under the **Apache 2.0 License**.  
See the full license here: [LICENSE](LICENSE)  

---

## Future Improvements  
- Support for handwritten text beyond digits  
- Enhancement in noise reduction and preprocessing  
- Deployment on cloud platforms  

If you find this repository useful, consider **starring** it! ⭐  
