import cv2
import numpy as np


cap = cv2.VideoCapture(0)

def reset():
	# ベース画像の作成に使う画像のフレーム数(枚数)
	frame_num = 30
	base_img = []
	for i in range(frame_num):
		ret, frame = cap.read()
		base_img += np.array(frame)

	# 各値の平均値を返す
	return base_img // frame_num


def detection(base_img, frame):
	color = [0, 0, 255]

	change_draw = [[[0, 0, 0] for w in range(frame.shape[x])] for h in range(frame.shape[y])]
	for h in range(frame.shape[y]):
		for w in range(frame.shape[x]):
			if abs(base_img[y][x][0] - frame[y][x][0]) > 10 or abs(base_img[y][x][1] - frame[y][x][1]) > 10 or abs(base_img[y][x][2] - frame[y][x][2]) > 10:
				change_draw[y][x] = color
	
	return change_draw


# ベース画像を用意
base_img = reset()

while True:
	ret, frame = cap.read()
	change_draw = detection(base_img, frame)

	# ベース画像に差分を書き加えた画像を作成
	edframe = np.array(base_img) + np.array(change_draw)

	# 加工済の画像を表示する
	cv2.imshow('Edited Frame', edframe)

	# キー入力を1ms待って、k が27（ESC）だったらBreakする
	k = cv2.waitKey(1)
	if k == 27:
		break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()