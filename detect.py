import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path):
    """Load the trained model from the given path."""
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    """Load and preprocess the Sudoku grid image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (450, 450))  
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    return thresh

def extract_digits(image):
    """Extract individual digit images from the Sudoku grid."""
    grid = []
    cell_size = image.shape[0] // 9  
    for i in range(9):
        row = []
        for j in range(9):
            x, y = j * cell_size, i * cell_size
            cell = image[y:y + cell_size, x:x + cell_size]
            cell = cv2.resize(cell, (28, 28))  
            cell = cell / 255.0  
            row.append(cell)
        grid.append(row)
    return np.array(grid).reshape(81, 28, 28, 1) 

def recognize_digits(model, digits):
    """Predict digits using the trained model."""
    predictions = model.predict(digits)
    predicted_numbers = [np.argmax(pred) if np.max(pred) > 0.5 else 0 for pred in predictions] 
    return np.array(predicted_numbers).reshape(9, 9)

def is_valid(board, row, col, num):
    """Check if placing num at board[row][col] is valid."""
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False

    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

def solve_sudoku(board):
    """Solve the Sudoku puzzle using backtracking."""
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if solve_sudoku(board):
                            return True
                        board[i][j] = 0
                return False
    return True

if __name__ == "__main__":
    model_path = "models/my_cnn_model.keras"  
    image_path = "sudoku_images/27.png"  

    model = load_model(model_path)
    processed_image = preprocess_image(image_path)
    digit_images = extract_digits(processed_image)
    sudoku_matrix = recognize_digits(model, digit_images)

    print("Recognized Sudoku Grid:")
    print(sudoku_matrix)

    if solve_sudoku(sudoku_matrix):
        print("Solved Sudoku Grid:")
        print(sudoku_matrix)
    else:
        print("No solution exists.")