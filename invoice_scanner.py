import cv2
from PIL import Image, ImageTk
from pytesseract import pytesseract
import tkinter as tk
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
        # Debugging: Print the raw extracted text
        print("Extracted text from image:")
        print(text)
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def detect_total_amount(text):
    # Adjust regex to handle different formats for amounts (including possible dollar sign, and no decimal part)
    total_amount = re.findall(r'\$?\s?(\d{1,3}(,\d{3})*(\.\d{2})?)', text)
    
    if total_amount:
        # Since re.findall returns tuples, we extract the matched number (first element in the tuple)
        return total_amount[0][0]
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

# Preprocess the image (convert to grayscale and apply thresholding)
def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)  # Read the image in grayscale
    if img is None:
        print("Error loading image.")
        return None
    _, thresh_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    preprocessed_path = "preprocessed_invoice.jpg"
    cv2.imwrite(preprocessed_path, thresh_img)
    return preprocessed_path

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
    else:
        print("Failed to read frame from the camera.")

def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)  # Read the image in grayscale
    if img is None:
        print("Error loading image.")
        return None
    # Remove thresholding to maintain the grayscale format without forcing black-and-white
    preprocessed_path = "preprocessed_invoice.jpg"
    cv2.imwrite(preprocessed_path, img)
    return preprocessed_path


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
    
    # Save the processed image
    preprocessed_path = "preprocessed_invoice.jpg"
    cv2.imwrite(preprocessed_path, thresh_img)
    
    return preprocessed_path

def detect_total_amount(text):
    # This regex handles multiple formats: with/without a dollar sign, commas, decimals, and spaces
    total_amount = re.findall(r'\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
    
    if total_amount:
        # Since re.findall returns a list of strings, we take the first matched number
        return total_amount[0]
    else:
        return "Total amount not found"

def extract_text(image_path):
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        # Debugging: Print the raw extracted text
        print("Extracted text from image:")
        print(text)  # Print the full raw text from the image
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

# Function to capture image, process, and display results
def capture_image():
    _, frame = camera.read()
    if frame is not None:
        print("Image captured successfully!")
        cv2.imwrite('invoice.jpg', frame)
        print("Image saved as 'invoice.jpg'.")

        # Preprocess the image
        preprocessed_image = preprocess_image('invoice.jpg')
        if preprocessed_image is None:
            print("Failed to preprocess the image.")
            return

        # Extract text from the preprocessed image
        text = extract_text(preprocessed_image)

        # Detect total amount and categorize the invoice
        detected_amount = detect_total_amount(text)
        invoice_category = categorize_invoice(text)

        # Set values in the GUI
        amount.set(detected_amount)
        category.set(invoice_category)

    else:
        print("Failed to capture image from the camera.")

# Set up the GUI
root = tk.Tk()
root.title("Invoice Scanner")

label = tk.Label(root)
label.pack()

capture_button = tk.Button(root, text="Capture", command=capture_image)
capture_button.pack()

amount = tk.StringVar()
category = tk.StringVar()

tk.Label(root, text="Total Amount:").pack()
tk.Label(root, textvariable=amount).pack()

tk.Label(root, text="Category:").pack()
tk.Label(root, textvariable=category).pack()

show_frame()
root.mainloop()
