import cognitive_face as CF
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import collections
import cv2
import math

KEY = '********************************'
BASE_URL = 'https://japaneast.api.cognitive.microsoft.com/face/v1.0'

CF.Key.set(KEY)
CF.BaseUrl.set(BASE_URL)

cascade_file = '/usr/local/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml'
cascade = cv2.CascadeClassifier(cascade_file)

img_url = 'photo.png'

# OpenCVの設定パラメータ
INTERVAL= 1
ESC_KEY = 27
ENTER_KEY = 13
DEVICE_ID = 0
WINDOW_NAME = "Facial muscle test"
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
GUIDE_WINDOW_NAME = "Guide"
GUIDE_WINDOW_WIDTH = 728
GUIDE_WINDOW_HEIGHT = 800



def scoring():
	age = 0
	gender = []
	score = []

	# FaceAPIで取得する感情
	attributes = ['anger', 'contempt', 'disgust', 'fear', 'smile', 'happiness', 'neutral', 'sadness', 'surprise']

	# 感情を提示し、それに応じた写真を送信
	for face in attributes:
		print(face)

		faces = []
		while faces == []:
			# ガイドの画像を表示
			cv2.imshow(GUIDE_WINDOW_NAME, cv2.resize(cv2.imread('img/{}.png'.format(face)), (GUIDE_WINDOW_WIDTH, GUIDE_WINDOW_HEIGHT)))
			
			# カメラ起動
			capture()

			# FaceAPIに投げる
			faces = CF.face.detect(img_url, face_id=True, landmarks=False, attributes='age,gender,smile,emotion')

			# 顔を認識できなかったときはやり直す
			if faces == []:
				cv2.imshow(GUIDE_WINDOW_NAME, cv2.resize(cv2.imread('img/fail.png'), (GUIDE_WINDOW_WIDTH, GUIDE_WINDOW_HEIGHT)))
				cv2.waitKey(0)


		# 顔を2つ以上認識したとき
		if len(faces) >= 2:
			face_width = 0
			for kao in faces:
				if kao['faceRectangle']['width'] > face_width:
					face_width = kao['faceRectangle']['width']
					temp_face = kao

			# 写真内で一番大きい顔を選択
			faces = temp_face

		
		# 笑顔のときだけ別処理
		if face == 'smile':
			score.append(round(faces[0]['faceAttributes']['smile'] * 100))
			print('{}\n'.format(faces[0]['faceAttributes']['smile']))
		else:
			score.append(round(faces[0]['faceAttributes']['emotion'][face] * 100))
			print('{}\n'.format(faces[0]['faceAttributes']['emotion']))

		age += faces[0]['faceAttributes']['age']
		gender.append(faces[0]['faceAttributes']['gender'])


	# 精度向上の為に9回の平均値を取る
	age = round(age // 9)

	# 精度向上の為に9回で最も多かった性別を選択する
	gender = collections.Counter(gender).most_common()[0][0]

	return score, age, gender


def plot_polar(values, gender):
	labels = ['怒り', '軽蔑', '嫌気', '恐怖', '笑顔', '幸福', '真顔', '悲しみ', '驚き']
	
	angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
	values = np.concatenate((values, [values[0]]))
	fig = plt.figure(figsize=(16, 12), dpi=100)
	ax = fig.add_subplot(111, polar=True)

	# 男性(青)
	if gender == 'male':
		ax.plot(angles, values, 'o-', linewidth=4)
		ax.fill(angles, values, alpha=0.25)

			
	# 女性(赤)
	elif gender == 'female':
		ax.plot(angles, values, 'mo-', linewidth=4)
		ax.fill(angles, values, 'r', alpha=0.25)


	# 点数を描画
	for idx, num in enumerate(values):
		ax.text(angles[idx], num, num, size=30, color='black', verticalalignment='center_baseline')

	ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, size=30)
	ax.set_rgrids(np.linspace(0, 100, 6), size=20, color='gray')

	fig.savefig('rader.png')
	plt.close(fig)


def capture():
	end_flag, c_frame = cap.read()

	while end_flag == True:
		img = c_frame
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))


		# 顔部分を赤枠で囲む
		for (x, y, w, h) in face_list:
			color = (0, 0, 225)
			pen_w = 3
			cv2.rectangle(c_frame, (x, y), (x+w, y+h), color, thickness = pen_w)


		# そのまま描画すると左右逆で混乱するので反転する
		c_frame = cv2.resize(cv2.flip(c_frame, 1), (WINDOW_WIDTH, WINDOW_HEIGHT))
		cv2.imshow(WINDOW_NAME, c_frame)
		key = cv2.waitKey(INTERVAL)


		# Escでプログラム終了
		if key == ESC_KEY:
			cv2.destroyAllWindows()
			cap.release()
			exit()


		# エンターキーで写真撮影
		elif key == ENTER_KEY:
			cv2.imwrite(img_url, c_frame)
			break

		end_flag, c_frame = cap.read()

	# 写真撮影から抜けるときはカメラを暗くする
	blended = cv2.addWeighted(c_frame, 0.4, mask, 0.3, 0)

	cv2.imshow(WINDOW_NAME, blended)
	cv2.imshow(GUIDE_WINDOW_NAME, cv2.resize(cv2.imread('img/wait.png'), (GUIDE_WINDOW_WIDTH, GUIDE_WINDOW_HEIGHT)))
	cv2.waitKey(1)

	

if __name__ == '__main__':
	# カメラとガイドの準備
	cap = cv2.VideoCapture(DEVICE_ID)
	cv2.namedWindow(WINDOW_NAME)
	cv2.moveWindow(WINDOW_NAME, 0, 0)
	cv2.namedWindow(GUIDE_WINDOW_NAME)
	cv2.moveWindow(GUIDE_WINDOW_NAME, 1024, 0)

	# レーダーチャート検証用
	gender = 'male'
	score = [19, 28, 37, 46, 55, 64, 73, 82, 91]
	
	# カメラ暗転用画像の用意
	ret, dummy = cap.read()
	mask = np.zeros_like(cv2.resize(cv2.flip(dummy, 1), (WINDOW_WIDTH, WINDOW_HEIGHT)))
	cv2.rectangle(mask, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (156, 156, 156), -1)


	while True:
		# スタート画面
		cv2.imshow(GUIDE_WINDOW_NAME, cv2.resize(cv2.imread('img/start.png'), (GUIDE_WINDOW_WIDTH, GUIDE_WINDOW_HEIGHT)))
		cv2.imshow(WINDOW_NAME, cv2.resize(cv2.imread('img/stand-by.png'), (WINDOW_WIDTH, WINDOW_HEIGHT)))
		cv2.waitKey(0)


		# 表情筋テスト開始
		score, age, gender = scoring()
		cv2.imshow(GUIDE_WINDOW_NAME, cv2.resize(cv2.imread('img/result.png'), (GUIDE_WINDOW_WIDTH, GUIDE_WINDOW_HEIGHT)))


		# レーダーチャート作成
		plot_polar(score, gender)
		cv2.imshow(WINDOW_NAME, cv2.resize(cv2.imread('rader.png'), (WINDOW_WIDTH, WINDOW_HEIGHT)))
		cv2.waitKey(0)


	# 終了処理
	cv2.destroyAllWindows()
	cap.release()


