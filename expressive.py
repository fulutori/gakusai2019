import cognitive_face as CF
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import collections
import random

KEY = '********************************'
BASE_URL = 'https://japaneast.api.cognitive.microsoft.com/face/v1.0'

CF.Key.set(KEY)
CF.BaseUrl.set(BASE_URL)

img_url = 'kousuke.jpg'

def scoring():
	age = 0
	gender = []
	score = []

	# FaceAPIで取得する感情
	attributes = ['anger', 'contempt', 'disgust', 'fear', 'smile', 'happiness', 'neutral', 'sadness', 'surprise']

	# 感情を提示し、それに応じた写真を送信
	for face in attributes:
		faces = CF.face.detect(img_url, face_id=True, landmarks=False, attributes='age,gender,smile,emotion')
		
		# 笑顔のときだけ別処理
		if face == 'smile':
			score.append(round(faces[0]['faceAttributes']['smile'] * 10))
		else:
			score.append(round(faces[0]['faceAttributes']['emotion'][face] * 10))

		age += faces[0]['faceAttributes']['age']
		gender.append(faces[0]['faceAttributes']['gender'])


	# 精度向上の為に9回の平均値を取る
	age = round(age // 9)

	# 精度向上の為に9回で最も多かった性別を選択する
	gender = collections.Counter(gender).most_common()[0][0]

	return score, age, gender


def plot_polar(values, gender, imgname):
	labels = ['怒り', '軽蔑', '嫌気', '恐怖','笑顔', '幸福', '真顔', '悲しみ', '驚き']
	
	angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
	values = np.concatenate((values, [values[0]]))
	fig = plt.figure()
	ax = fig.add_subplot(111, polar=True)

	# 男性と女性でグラフの色を変える
	if gender == 'male': #青
		ax.plot(angles, values, 'o-')
		ax.fill(angles, values, alpha=0.25)
	elif gender == 'female': #赤
		ax.plot(angles, values, 'mo-')
		ax.fill(angles, values, 'r', alpha=0.25)

	ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
	ax.set_rlim(0 ,10)

	fig.savefig(imgname)
	plt.close(fig)


if __name__ == '__main__':
	score, age, gender = scoring()
	plot_polar(score, gender, 'test.png')
