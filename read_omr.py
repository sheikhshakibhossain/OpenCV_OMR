import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def order_points(pts):
    """Order points in clockwise order starting from top-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left will have the smallest sum
    # Bottom-right will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right will have smallest difference
    # Bottom-left will have largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    """Apply perspective transform to get top-down view"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Create destination points for transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Compute perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def grab_contours(cnts):
    """Handle different versions of OpenCV findContours return value"""
    if len(cnts) == 2:
        return cnts[0]
    elif len(cnts) == 3:
        return cnts[1]
    else:
        return cnts

def process_omr_sheet(image_path):
    """Process an OMR sheet image and extract registration number and answers"""
    # Step 1: Load the image, make a copy for visualization, and convert to grayscale
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Edge detection
    edged = cv2.Canny(blurred, 75, 200)
    
    # Step 4: Find contours to identify the OMR sheet
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    
    # Sort contours by size and keep the largest ones
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    omr_contour = None
    
    # Find the contour with 4 points (our OMR sheet should be rectangular)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            omr_contour = approx
            break
    
    if omr_contour is None:
        print("Could not find OMR sheet contour")
        return None
    
    # Step 5: Apply perspective transform to get a top-down view of the OMR sheet
    paper = four_point_transform(image, omr_contour.reshape(4, 2))
    warped_gray = four_point_transform(gray, omr_contour.reshape(4, 2))
    
    # Step 6: Apply thresholding to get a binary image
    thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Display the processed image (for debugging)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
    plt.title("Thresholded OMR Sheet")
    plt.axis('off')
    plt.show()
    
    # Step 7: Define regions for registration number and answers
    # Note: These coordinates need to be adjusted based on your specific OMR sheet
    height, width = thresh.shape
    
    # Define registration number region | adjust
    reg_no_region = thresh[int(height*0.179):int(height*0.39), int(width*0.01):int(width*0.264)]
    
    # Define answers region | adjust
    answers_region = thresh[int(height*0.675):int(height*0.95), int(width*0.0):int(width*0.99)]
    
    # Display these regions for verification
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(reg_no_region, cmap='gray')
    plt.title("Registration Number Region")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(answers_region, cmap='gray')
    plt.title("Answers Region")
    plt.axis('off')
    plt.show()
    
    # Step 8: Extract registration number by detecting filled bubbles
    reg_no = extract_registration_number(reg_no_region)
    
    # Step 9: Extract answers by detecting filled bubbles
    answers = extract_answers(answers_region)
    
    return {
        "registration_number": reg_no,
        "answers": answers
    }

def extract_registration_number(reg_no_region):
    """Extract registration number from the bubble grid"""
    # Need to detect the grid structure of the reg no area
    # This function would detect filled bubbles in each column and convert to a digit
    
    # For now, we'll implement a simplified version that assumes 10 columns, each with 10 digits (0-9)
    h, w = reg_no_region.shape
    
    # Estimate column width
    col_width = w // 10
    
    reg_no = ""
    
    # Process each column
    for col in range(10):
        x_start = col * col_width
        x_end = (col + 1) * col_width
        
        column = reg_no_region[:, x_start:x_end]
        
        # Divide the column into 10 equal parts (for digits 0-9)
        row_height = h // 10
        max_pixel_count = 0
        selected_digit = -1
        
        for digit in range(10):
            y_start = digit * row_height
            y_end = (digit + 1) * row_height
            
            bubble = column[y_start:y_end, :]
            pixel_count = cv2.countNonZero(bubble)
            
            if pixel_count > max_pixel_count:
                max_pixel_count = pixel_count
                selected_digit = digit
        
        # Only add digit if we found a marked bubble (threshold can be adjusted)
        if max_pixel_count > (row_height * col_width * 0.3):
            reg_no += str(selected_digit)
        else:
            reg_no += "_"  # No bubble marked for this column
    
    return reg_no

def extract_answers(answers_region):
    """Extract answers (A, B, C, D) from the bubble grid"""
    # For this function, we need to:
    # 1. Detect the rows for each question
    # 2. For each row, detect which bubble is filled (A, B, C, D)
    
    # This is a simplified implementation
    h, w = answers_region.shape
    
    # Let's assume 40 questions (10 rows with 4 columns in each of 4 sections)
    # This needs to be customized based on the actual layout
    
    # First, let's identify the four sections (blocks of questions)
    section_width = w // 4
    
    answers = {}
    question_number = 1
    
    # Process each section
    for section in range(4):
        section_x_start = section * section_width
        section_x_end = (section + 1) * section_width
        
        section_region = answers_region[:, section_x_start:section_x_end]
        section_h, section_w = section_region.shape
        
        # Each section has 10 questions
        row_height = section_h // 10
        
        # Process each question (row)
        for row in range(10):
            row_start = row * row_height
            row_end = (row + 1) * row_height
            
            question_row = section_region[row_start:row_end, :]
            
            # Each question has 4 options (A, B, C, D)
            option_width = section_w // 5  # 5 parts - one for question number, 4 for options
            
            max_pixel_count = 0
            selected_option = None
            
            # Check each option
            for option in range(4):
                option_start = (option + 1) * option_width  # +1 to skip question number column
                option_end = (option + 2) * option_width
                
                bubble = question_row[:, option_start:option_end]
                pixel_count = cv2.countNonZero(bubble)
                
                if pixel_count > max_pixel_count:
                    max_pixel_count = pixel_count
                    selected_option = chr(65 + option)  # Convert to A, B, C, D
            
            # Add to answers dictionary if we found a marked bubble
            if max_pixel_count > (row_height * option_width * 0.3):
                answers[question_number] = selected_option
            else:
                answers[question_number] = None  # No bubble marked for this question
            
            question_number += 1
    
    return answers

def visualize_results(image_path, results):
    """Visualize the extracted results on the original image"""
    # Load the original image
    image = cv2.imread(image_path)
    
    # Create a copy for visualization
    viz_image = image.copy()
    
    # Add text for registration number
    cv2.putText(viz_image, f"Reg No: {results['registration_number']}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Add text for first few answers (as an example)
    y_pos = 70
    for q in range(1, min(6, len(results['answers']) + 1)):
        ans_text = f"Q{q}: {results['answers'][q]}" if results['answers'][q] else f"Q{q}: NONE"
        cv2.putText(viz_image, ans_text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
    
    # Display more answers with ...
    if len(results['answers']) > 5:
        cv2.putText(viz_image, "...", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))
    plt.title("OMR Results Visualization")
    plt.axis('off')
    plt.show()

def main():
    """Main function to process an OMR sheet"""
    # Path to your OMR sheet image
    # image_path = "omr.png"
    image_path = str(sys.argv[1]).strip()
    
    # Process the OMR sheet
    results = process_omr_sheet(image_path)
    
    if results:
        print("Registration Number:", results["registration_number"])
        print("\nAnswers:")
        for question, answer in results["answers"].items():
            print(f"Question {question}: {answer}")
        
        # Visualize results
        visualize_results(image_path, results)

if __name__ == "__main__":
    main()