import time
import cv2
from imgnet import ImgNet

def capture_camera(mirror=False, size=None, dev_num=0):
    """Capture video from camera"""

    cap = cv2.VideoCapture(dev_num)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    img_net = ImgNet("mobilenet")

    flag_1st = False
    while True:
        # retは画像を取得成功フラグ
        start = time.time()
        ret, frame = cap.read()
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
        
        target_array = frame[xp:xp+448,yp:yp+448,:]
        results = img_net.array2predict(cv2.resize(target_array,(224,224)))
        tag = "{0}_{1:.3f}".format(results[0][1], results[0][2])
        print(tag)
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

    # カメラを変える場合は引数を 0 --> 1
    capture_camera(0)
