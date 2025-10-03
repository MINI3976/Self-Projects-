

import os
import time

def read_board(file_path):
    board = []
    with open(file_path, "r") as f:
        for line in f:
            board.append([int(num) for num in line.split()])
    return board

def create_sample_board(file_path):
    sample_board = """5 3 0 0 7 0 0 0 0
6 0 0 1 9 5 0 0 0
0 9 8 0 0 0 0 6 0
8 0 0 0 6 0 0 0 3
4 0 0 8 0 3 0 0 1
7 0 0 0 2 0 0 0 6
0 6 0 0 0 0 2 8 0
0 0 0 4 1 9 0 0 5
0 0 0 0 8 0 0 7 9"""
    with open(file_path, "w") as f:
        f.write(sample_board)

def print_board(board):
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - -")
        for j in range(len(board[0])):
            if j % 3 == 0 and j != 0:
                print("| ", end="")
            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")

def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # row, col
    return None

def is_valid(board, num, pos):
    row, col = pos

    # Check row
    for j in range(len(board[0])):
        if board[row][j] == num and j != col:
            return False

    # Check column
    for i in range(len(board)):
        if board[i][col] == num and i != row:
            return False

    # Check box
    box_x = col // 3
    box_y = row // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x*3, box_x*3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True

def solve(board):
    find = find_empty(board)
    if not find:
        return True  # Solved
    else:
        row, col = find

    for num in range(1, 10):
        if is_valid(board, num, (row, col)):
            board[row][col] = num

            if solve(board):
                return True

            board[row][col] = 0  # Reset

    return False

if __name__ == "__main__":
    board_file = "sudoku_board.txt"

    if not os.path.exists(board_file):
        print(f"{board_file} not found. Creating a sample Sudoku board...")
        create_sample_board(board_file)

    board = read_board(board_file)

    print("Initial Sudoku:")
    print_board(board)

    start_time = time.time()
    if solve(board):
        print("\nSolved Sudoku:")
        print_board(board)
    else:
        print("No solution exists!")

    print(f"\nTime taken: {time.time() - start_time:.4f} seconds")
