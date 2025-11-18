import cv2
import pytesseract
import pandas as pd
import re

CATEGORY_RULES = {
    "Jedzenie": ["wędlina", "masło"],
    "Transport": ["bilet", "paliwo"]
}

def assign_category(item_name: str) -> str:
    text = item_name.lower()
    for category, keywords in CATEGORY_RULES.items()
        if any(k in text for k in keywords):
            return category
    return "Inna"

def preprocess_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def run_ocr(image):
    return pytesseract.image_to_string(image)

LINE_ITEM_REGEX = re.comile(r"(.+?)\s+(\d+)\s+(\d+[\.,]\d{2}$")

def parse_items(ocr_text):
    item = []
    for line in ocr_text.split("\n"):
        match = LINE_ITEM_REGEX.search(line.strip())
        if not match:
            continue
        name, qty, price = match.groups()
        price = float(price.replace(",", "."))
        qty = int(qty)
        total = round(qty * price, 2)
        category = assign_category(name)
        items.append([name, qty, price, total, category])
    return items

def extract_receipt(path: str):
    image = preprocess_image(path)
    text = run_ocr(image)
    items = parse_items(text)

    df = pd.DataFrame(items, columns=["Item", "Qty", "Price", "Total", "Category"])
    return df

if __name__ = "__main__":
    receipt_path = "example.jpg"
    df = extract_receipt(receipt_path)
    print(df)
    df.to_csv("receipt_parsed.csv", index=False)
    print("Saved to receipt_parsed.csv")

