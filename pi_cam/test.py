import cv2

# 画像の読み取り
img = cv2.imread("./test.jpg")

print(img.shape)

# グレースケール変換
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# リサイズ
img_resize = cv2.resize(img, (300, 300))

# 文字のプリント
img_resize = cv2.putText(img_resize, 'zou', (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), thickness=2)

# 画像のアウトプット
cv2.imshow("BGR", img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()