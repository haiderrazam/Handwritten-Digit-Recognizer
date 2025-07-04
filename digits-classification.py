import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Multi-Digit Recognizer", layout="centered")
st.title("🔢 Multi-Digit Recognition")

# Load trained model
model = tf.keras.models.load_model("mnist_cnn.h5")

def preprocess_and_predict_digit(digit_img):
    # Make square canvas around digit
    h, w = digit_img.shape
    size = max(w, h)
    square_canvas = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square_canvas[y_offset:y_offset + h, x_offset:x_offset + w] = digit_img

    # Resize to MNIST 28x28 and normalize
    resized = cv2.resize(square_canvas, (28,28), interpolation=cv2.INTER_AREA)
    norm_img = resized.astype("float32") / 255.0
    input_img = norm_img.reshape(1,28,28,1)

    pred = model.predict(input_img, verbose=0)
    return pred.argmax(axis=1)[0]

st.write("### Upload an image containing multiple handwritten digits in a line:")
uploaded_file = st.file_uploader("Upload PNG/JPG image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.write("### Uploaded Image")
    st.image(image, caption="Original uploaded image", channels="GRAY")

    # Threshold image: black background, white digits
    _, img_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ##st.write("### Thresholded Image")
    ##st.image(img_thresh, caption="Thresholded", channels="GRAY")

    # Find contours (digits)
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Sort contours left to right
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        contours_sorted = [c for _, c in sorted(zip(bounding_boxes, contours), key=lambda b: b[0][0])]

        predicted_digits = []
        boxed_img = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)

        for cnt in contours_sorted:
            x, y, w, h = cv2.boundingRect(cnt)
            digit_crop = img_thresh[y:y+h, x:x+w]

            predicted_digit = preprocess_and_predict_digit(digit_crop)
            predicted_digits.append(str(predicted_digit))

            # Draw rectangle on boxed_img for visualization
            cv2.rectangle(boxed_img, (x, y), (x+w, y+h), (0,255,0), 2)

        recognized_number = "".join(predicted_digits)
        st.write("### Detected Digits (in order):")
        st.success(f"🔢 Recognized number: **{recognized_number}**")

        st.image(boxed_img, caption="Digits with bounding boxes", channels="BGR")
    else:
        st.warning("No digits detected. Please upload an image with clear digits.")
