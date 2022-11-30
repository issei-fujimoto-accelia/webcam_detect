# readme
動作環境

```
$ python --version
Python 3.9.5
```

## install
`pip install -r requirements.txt`

## train
run: `python train.py {epoch}`

データセットは`./dataset/d.json`を読み込みます。

d.jsonの例:  
https://drive.google.com/drive/u/1/folders/1PC5i_jfWJTBnKsk2nBxEFgIipvgbqeRP


```
{"image_path": "./dataset/raw/000041.jpg", "labels": {"class_labels": [0, 0, 0, 0, 0], "boxes": [[0.477, 0.5265, 0.196, 0.213], [0.654, 0.634, 0.206, 0.196], [0.8075, 0.7275, 0.207, 0.215], [0.293, 0.767, 0.2, 0.178], [0.49, 0.705, 0.164, 0.154]]}}
```


ディレクトリは以下のようになります。

- train.py
  - dataset/
	- d.json
	- raw/
		- 001.jpg
		- 002.jpg
		


学習を実行させたcolab  
https://colab.research.google.com/drive/1TdE8vuQThHcylsjiQ_PuIFhcn4SVKyHI?authuser=1

mountするdriveは、d.jsonと蕪の画像を含む必要があります。

### 学習済みモデル
蕪を学習させたweight  
https://drive.google.com/drive/u/1/folders/18Gj-TAKg707IyJVUQ0hA6RyD_9XVc8Aq

保存済みモデルを使う場合、このweightをダウンロードしてきて`./models/`以下など任意の位置に配置してください。

### dataset
`crawler.py`  
蕪の画像を集めてくる.  
保存dirとか取ってくる画像の数を変更するには`name("蕪", "kabu", 30)`を変更する  


`crawler.py`  
`./dataset/label`と`./dataset/raw`を元に、train用のjsonファイルを作成します.  


`rename.py`  
指定したディレクトリのファイルを指定したnumberでrenameしなおす。


## webcamを使ったdetect
run: `python main.py`

webcamの変更はこの辺を書き換える。webcamは1のことが多いが、0の場合もある


```
cap = cv2.VideoCapture(0)
```


カメラに合わせて解像度を変える場合はこの辺を書き換える

```
WIDTH=1280
HEIGHT=720
```


### 保存済みモデルの利用
`./detect_yolo.py`の`self._model = YolosForObjectDetection.from_pretrained('./models/20221117/')`  
を保存したモデルのpathに書き換えてください.

## 画像を使ったテスト
`python detect_for_img.py`

画像に対してdetectのテストを行います。

テストする画像は、ここを手動で書き換えてください  
`image = Image.open("./images/kabu3.jpeg")`
