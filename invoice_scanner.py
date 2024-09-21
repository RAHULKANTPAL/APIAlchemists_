import cv2
from PIL import Image, ImageTk
from pytesseract import pytesseract
import tkinter as tk
from tkinter import ttk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Set up the video capture
camera = cv2.VideoCapture(0)

# OCR and machine learning functionality
def extract_text(image_path):
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def detect_total_amount(text):
    total_amount = re.findall(r'\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
    if total_amount:
        return total_amount[0]
    else:
        return "Total amount not found"

# Machine learning model setup
def train_model():
    data = [
        ("Electricity bill from XYZ", "Utility"),
        ("Invoice for office supplies", "Office Supplies"),
        ("Rent payment", "Rent"),
    ]
    texts, labels = zip(*data)
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
    model.fit(X_train, y_train)
    return model

model = train_model()

def categorize_invoice(text):
    return model.predict([text])[0]

def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)  # Read the image in grayscale
    if img is None:
        print("Error loading image.")
        return None
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Adaptive thresholding for better text detection
    thresh_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    
    preprocessed_path = "preprocessed_invoice.jpg"
    cv2.imwrite(preprocessed_path, thresh_img)
    
    return preprocessed_path

def capture_image():
    _, frame = camera.read()
    if frame is not None:
        cv2.imwrite('invoice.jpg', frame)
        
        # Preprocess the image
        preprocessed_image = preprocess_image('invoice.jpg')
        if preprocessed_image is None:
            return
        
        # Extract text from the preprocessed image
        text = extract_text(preprocessed_image)
        
        # Detect total amount and categorize the invoice
        detected_amount = detect_total_amount(text)
        invoice_category = categorize_invoice(text)
        
        # Set values in the GUI
        amount.set(f"Total: {detected_amount}")
        category.set(f"Category: {invoice_category}")

# GUI functionality to display video feed
def show_frame():
    _, frame = camera.read()
    if frame is not None:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.after(10, show_frame)

# Set up the GUI
root = tk.Tk()
root.title("Invoice Scanner")
root.geometry("600x700")

# Set up a style for the ttk widgets
style = ttk.Style()
style.configure("TLabel", font=("Arial", 12), padding=10)
style.configure("TButton", font=("Arial", 12), padding=5)
style.configure("TFrame", background="#BBE9FF", padding=10)

# Main Frame for layout
main_frame = ttk.Frame(root)
main_frame.pack(expand=True)

# Video feed section
video_frame = ttk.LabelFrame(main_frame, text="Live Video Feed")
video_frame.grid(row=0, column=0, padx=10, pady=10)

label = ttk.Label(video_frame)
label.pack()

# Control section
control_frame = ttk.Frame(main_frame)
control_frame.grid(row=1, column=0, padx=10, pady=20)

capture_button = ttk.Button(control_frame, text="Capture Invoice", command=capture_image)
capture_button.pack(pady=10)

# Results section
result_frame = ttk.LabelFrame(main_frame, text="Detected Information")
result_frame.grid(row=2, column=0, padx=10, pady=10)

amount = tk.StringVar()
category = tk.StringVar()

# Align the result information in the center
ttk.Label(result_frame, textvariable=amount, font=("Arial", 14), anchor="center").pack(pady=10)
ttk.Label(result_frame, textvariable=category, font=("Arial", 14), anchor="center").pack(pady=10)

# Make everything centered
main_frame.grid_columnconfigure(0, weight=1)
result_frame.grid_columnconfigure(0, weight=1)
control_frame.grid_columnconfigure(0, weight=1)
video_frame.grid_columnconfigure(0, weight=1)

# Start video feed
show_frame()

root.mainloop()
