import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import os
import xlwings as xw
from datetime import datetime

# Initialize PaddleOCR
ocr = PaddleOCR()
cap = cv2.VideoCapture('vid.mp4')
model = YOLO("best.pt")
with open("coco1.txt", "r") as f:
    class_names = f.read().splitlines()
def save_to_excel(detected_text):
    # Get the current date and time
    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%Y-%m-%d")
    current_time = current_datetime.strftime("%H-%M")  # Use minutes for the filename

    # Define the filename for the workbook with the current date
    filename = f"{current_date}_detected_plates.xlsx"
    
    # Open the workbook
    if os.path.exists(filename):
        wb = xw.Book(filename)  # Open the existing workbook
    else:
        wb = xw.Book()  # Create a new workbook
        wb.save(filename)  # Save it with the filename

    # Check if a sheet for the current date already exists
    sheets = wb.sheets
    sheet_names = [sheet.name for sheet in sheets]
    
    if current_date in sheet_names:
        sheet = wb.sheets[current_date]  # Use the existing sheet
    else:
        sheet = wb.sheets.add(current_date)  # Create a new sheet for the date

    # Find the next empty row by counting non-empty cells in column A
    col_a = sheet.range('A:A').value  # Get all values in column A
    next_row = len([cell for cell in col_a if cell is not None]) + 1

    # Write the detected text, current date, and time
    sheet.range(f'A{next_row}').value = detected_text  # Column A: Detected text
    sheet.range(f'B{next_row}').value = current_date   # Column B: Current date
    sheet.range(f'C{next_row}').value = current_time    # Column C: Current time

    # Adjust column widths to fit the content
    sheet.range('A:C').autofit()  # Auto-adjust columns A to C

    # Format column B for dates if needed
    sheet.range(f'B{next_row}').number_format = 'yyyy-mm-dd'

    # Optionally save the workbook after changes
    wb.save()

# Function to perform OCR on an image array
def perform_ocr(image_array):
    if image_array is None:
        raise ValueError("Image is None")

    # Perform OCR on the image array
    results = ocr.ocr(image_array, rec=True)  # rec=True enables text recognition
    detected_text = []

    # Process OCR results
    if results[0] is not None:
        for result in results[0]:
            text = result[1][0]
            detected_text.append(text)
      
    # Join all detected texts into a single string
    return ''.join(detected_text)


# Mouse callback function to print mouse position
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Initialize video capture and YOLO model
count = 0
area = [(612, 316), (598, 365), (938, 344), (924, 307)]
counter = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            
            result = cv2.pointPolygonTest(np.array(area, np.int32), (x1, y1), False)
            if result >= 0:
                if track_id not in counter:
                    counter.append(track_id)  # Only add if it's a new track ID
                    crop = frame[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (110, 70))
                    text = perform_ocr(crop)
                    text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '').replace('-', ' ')
                    
                    # Save detected text to Excel
                    save_to_excel(text)

    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
