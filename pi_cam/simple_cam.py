import time
import cv2


# カメラから画像を取得して表示 文字などを追加するだけのプログラム
def capture_camera(mirror=False, size=None, dev_num=0):
    """Capture video from camera"""

    cap = cv2.VideoCapture(dev_num)

    flag_1st = False
    while True:
        # retは画像を取得成功フラグ
        start = time.time()
        ret, frame = cap.read()

        # 鏡のように映るか否か
        if mirror is True:
            frame = frame[:,::-1]

        # フレームをリサイズ
        # sizeは例えば(800, 600)
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        if flag_1st == False:
            xp = int((frame.shape[1] - 448 ) / 2)
            yp = int((frame.shape[0] - 448 ) / 2)
            flag_1st = True
        # print(frame.shape)
        frame = cv2.rectangle(frame, (xp, yp), (xp+448, yp+448), (0, 0, 255), lineType=cv2.LINE_AA, thickness=4)
        
        tag = "simpple_cam"
        frame = cv2.putText(frame, tag, (xp+ 5, yp+25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), thickness=2)
        # フレームを表示する
        cv2.imshow('camera capture', frame)
        
        elapsed_time = time.time() - start
        print ("elapsed_time:{0:.3f}".format(elapsed_time) + "[sec]")

        k = cv2.waitKey(1) # 1msec待つ
        if k == 27: # ESCキーで終了
            break

    # キャプチャを解放する
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # ダメなら中の数字を1に変更
    capture_camera(0)
