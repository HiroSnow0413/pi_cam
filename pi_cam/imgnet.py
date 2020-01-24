


from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
import numpy as np

class ImgNet(object):

    def __init__(self, name="resnet"):

        
        # 学習済みのVGG16をロード
        # 構造とともに学習済みの重みも読み込まれる
        if name == "mobilenet":
            self.model = MobileNet(weights='imagenet')
        elif name == "resnet":
            self.model = ResNet50(weights='imagenet')
        else:
            self.model = ResNet50(weights='imagenet')
        self.model.summary()


    def array2predict(self, x):

        # 3次元テンソル（rows, cols, channels) を
        # 4次元テンソル (samples, rows, cols, channels) に変換
        # 入力画像は1枚なのでsamples=1でよい
        x = np.expand_dims(x, axis=0)

        # Top-5のクラスを予測する
        # VGG16の1000クラスはdecode_predictions()で文字列に変換される
        preds = self.model.predict(preprocess_input(x))
        results = decode_predictions(preds, top=5)[0]
        
        return results

    def file2predict(self, filename):

        # 引数で指定した画像ファイルを読み込む
        # サイズはVGG16のデフォルトである224x224にリサイズされる
        img = image.load_img(filename, target_size=(224, 224))
        # 読み込んだPIL形式の画像をarrayに変換
        x = image.img_to_array(img)

        results = self.array2predict(x)

        return results


if __name__ == '__main__':

    img_net = ImgNet("resnet")

    results = img_net.file2predict("./test.jpg")
    
    for result in results:
        print(result)
    print("{0}_{1:.3f}".format(results[0][1], results[0][2]))