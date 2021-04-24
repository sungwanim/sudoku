import numpy as np
import cv2
import pytesseract
from PIL import Image
import re
import sys

img = cv2.imread("/Users/sungwanim/Downloads/sudoku.png")
width = img.shape[0]
height = img.shape[1]

div = 0.0
cols_ratio = 0.0
rows_ratio = 0.0
if img.shape[0] > img.shape[1]:
    div = img.shape[0] / img.shape[1]
    cols_ratio = 1.0
    rows_ratio = 1.0 / div
else:
    cols_ratio = img.shape[1] / img.shape[0]
    rows_ratio = 1.0
img = cv2.resize(img, (int(2000 * cols_ratio), int(2000 * rows_ratio)))



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (0, 0), 3)
thresh = cv2.adaptiveThreshold(gray, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 5)


edged = cv2.Canny(thresh, 50, 150)

lines = cv2.HoughLines(edged, 1, np.pi / 180, 300)

pts = []
for l in lines:
    r, t = l[0]
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    x0 = r * cos_t
    y0 = r * sin_t
    alpha = 2500

    pt1 = (np.round(x0 + alpha * (-sin_t)), np.round(y0 + alpha * cos_t))
    pt2 = (np.round(x0 - alpha * (-sin_t)), np.round(y0 - alpha * cos_t))
    middle = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
    pts.append(middle)

pts.sort(key=lambda pts: pts[0])
sortx = []
cnt = 0
for i in pts:
        # x좌표가 0보다 클 때
    if i[0] > 0:
        if cnt == 0:
            sortx.append(i)
            cnt += 1
        # 바로 앞 점의 x좌표와 현재 점의 x좌표가 60 이상 차이날 때
        elif i[0] - sortx[-1][0] > 60:
            sortx.append(i)
            cnt += 1

pts.sort(key=lambda pts: pts[1])
sorty = []
cnt = 0

for i in pts:
        # y좌표가 0보다 클 때
    if i[1] > 0:
        if cnt == 0:
            sorty.append(i)
            cnt += 1
        # 바로 앞 점의 y좌표와 현재 점의 y좌표가 60 이상 차이날 때
        elif i[1] - sorty[-1][1] > 60:
            sorty.append(i)
            cnt += 1

coords = np.full((10, 10, 2), (0, 0))

for i in range(10):
    for j in range(10):
        coords[i, j] = (sortx[i][0], sorty[j][1])

sudoku = np.zeros((9, 9))


zeros = []
Row = [[False] * 10] * 9
Col = [[False] * 10] * 9
Square = [[False] * 10] * 9

p = re.compile("[\d]")

middlePoint = np.zeros((9, 9, 2))
for i in range(9):
    for j in range(9):
        leftTop = coords[j, i]
        rightBottom = coords[j + 1, i + 1]
        middlePoint[i][j][0] = (rightBottom[0] + leftTop[0]) / 2
        middlePoint[i][j][1] = (rightBottom[1] + leftTop[1]) / 2
        slicedImg = Image.fromarray(thresh[leftTop[0] + 40:rightBottom[0] - 40 + 1, leftTop[1] + 40:rightBottom[1] - 40 + 1])
        imgString = pytesseract.image_to_string(slicedImg, config="-c tessedit_char_whitelist=0123456789 --psm 6")
        m = p.match(imgString)
        if m:
            sudoku[j][i] = m.group()[0]
        else:
            sudoku[j][i] = 0
            zeros.append([j, i])

def is_possible(x, y, num):
    if sudoku[x][y] != 0:
        return False
    for tmp in range(9):
        if num == sudoku[x][tmp]:
            return False
    for tmp in range(9):
        if num == sudoku[tmp][y]:
            return False
    boxX = x // 3
    boxY = y // 3
    for i in range(boxX * 3, (boxX + 1) * 3):
        for j in range(boxY * 3, (boxY + 1) * 3):
            if sudoku[i, j] == num:
                return False
    return True

end = False
answer = np.zeros((9, 9))

def dfs(cnt):
    global end
    global answer

    if end:
        return

    if cnt == len(zeros):
        answer = sudoku.copy()
        end = True
        return
    for i in range(1, 10):
        col = zeros[cnt][0]
        row = zeros[cnt][1]
        if is_possible(col, row, i):
            sudoku[col, row] = i
            dfs(cnt + 1)
            sudoku[col, row] = 0

dfs(0)
for i in range(9):
    for j in range(9):
        if sudoku[i][j] == 0:
            leftBottom = [int(middlePoint[i][j][0]) - 50, int(middlePoint[i][j][1]) + 50]
            cv2.putText(img, str(int(answer[i][j])), (leftBottom[0], leftBottom[1]), cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 2, cv2.LINE_AA)
img = cv2.resize(img, (width, height), cv2.INTER_AREA)
cv2.imshow("sudoku", img)
cv2.waitKey(0)
