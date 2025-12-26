"""Analyze certificate image quality and test different OCR preprocessing strategies"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
import cv2
import numpy as np
import pytesseract

# Configure tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def analyze_image(img_path):
    """Analyze image quality and resolution"""
    print(f"\n{'='*70}")    
    print(f"📸 Analyzing: {os.path.basename(img_path)}")
    print(f"{'='*70}")
    
    # Load with PIL
    pil_img = Image.open(img_path)
    print(f"📏 Image Size: {pil_img.size[0]}x{pil_img.size[1]} pixels")
    print(f"📄 Mode: {pil_img.mode}")
    print(f"📊 Format: {pil_img.format}")
    
    # Convert to numpy array
    img = np.array(pil_img)
    
    # Check if image is already grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Calculate image statistics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    print(f"💡 Mean Brightness: {mean_brightness:.2f}/255")
    print(f"📊 Brightness Std Dev: {std_brightness:.2f}")
    print(f"🎯 Contrast Ratio: {std_brightness/mean_brightness:.3f}")
    
    return img, gray


def test_preprocessing_strategies(img_path):
    """Test different OCR preprocessing strategies"""
    img, gray = analyze_image(img_path)
    
    strategies = {
        "1. Direct OCR (no preprocessing)": lambda: pytesseract.image_to_string(gray),
        
        "2. Binary Threshold (127)": lambda: pytesseract.image_to_string(
            cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        ),
        
        "3. Adaptive Threshold (Gaussian)": lambda: pytesseract.image_to_string(
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        ),
        
        "4. OTSU Threshold": lambda: pytesseract.image_to_string(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        ),
        
        "5. Denoise + OTSU": lambda: pytesseract.image_to_string(
            cv2.threshold(
                cv2.fastNlMeansDenoising(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
        ),
        
        "6. Upscale 2x + OTSU": lambda: pytesseract.image_to_string(
            cv2.threshold(
                cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
        ),
        
        "7. Sharpen + OTSU": lambda: pytesseract.image_to_string(
            cv2.threshold(
                cv2.filter2D(gray, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])),
                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
        ),
    }
    
    print(f"\n{'='*70}")
    print("🔬 Testing OCR Preprocessing Strategies")
    print(f"{'='*70}")
    
    results = {}
    for name, strategy in strategies.items():
        print(f"\n{name}")
        print("-" * 70)
        try:
            text = strategy()
            char_count = len(text.strip())
            word_count = len(text.split())
            
            # Check for key indicators
            has_grades = bool(re.search(r'\b[A-F][1-9]\b', text))
            has_exam_num = bool(re.search(r'\d{7,}', text))
            
            print(f"  ✓ Characters: {char_count}")
            print(f"  ✓ Words: {word_count}")
            print(f"  ✓ Has Grades: {'✅' if has_grades else '❌'}")
            print(f"  ✓ Has Exam Number: {'✅' if has_exam_num else '❌'}")
            
            # Show a snippet
            snippet = text[:200].replace('\n', ' ')
            print(f"  📝 Snippet: {snippet}...")
            
            results[name] = {
                'text': text,
                'char_count': char_count,
                'word_count': word_count,
                'has_grades': has_grades,
                'has_exam_num': has_exam_num
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[name] = {'error': str(e)}
    
    return results


import re

if __name__ == "__main__":
    # Test on forged certificates
    test_images = [
        "training_data/val/forged/forged_0001.jpg",
        "training_data/val/forged/forged_0003.jpg",
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            results = test_preprocessing_strategies(img_path)
            
            # Find best strategy
            print(f"\n{'='*70}")
            print("🏆 Best Strategy Analysis")
            print(f"{'='*70}")
            
            best = None
            best_score = 0
            for name, result in results.items():
                if 'error' not in result:
                    score = (
                        result['char_count'] * 0.3 +
                        result['word_count'] * 1.0 +
                        result['has_grades'] * 50 +
                        result['has_exam_num'] * 30
                    )
                    if score > best_score:
                        best_score = score
                        best = name
            
            print(f"🥇 Winner: {best}")
            print(f"📊 Score: {best_score:.2f}")
        else:
            print(f"❌ File not found: {img_path}")
