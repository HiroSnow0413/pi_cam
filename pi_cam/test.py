import cv2

# �摜�̓ǂݎ��
img = cv2.imread("./test.jpg")

print(img.shape)

# �O���[�X�P�[���ϊ�
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ���T�C�Y
img_resize = cv2.resize(img, (300, 300))

# �����̃v�����g
img_resize = cv2.putText(img_resize, 'zou', (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), thickness=2)

# �摜�̃A�E�g�v�b�g
cv2.imshow("BGR", img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()