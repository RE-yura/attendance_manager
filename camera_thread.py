#!/usr/bin/python3
# coding: utf-8
import cv2
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtTest import QTest

from face_learning import FaceLearner

# === カメラ処理用スレッド ==============================
class CameraThread(QThread):
  changePixmap = pyqtSignal(QImage)
  changeLabel = pyqtSignal(str)
  changeProgress = pyqtSignal(int)

  #【カメラモードの切り替え】
  # 0 : 内蔵カメラ
  # 1 : USBカメラ
  CAMERA_MODE = 0

  def __init__(self):
    super().__init__()
    self.fc = FaceLearner()
    self.fc.changeProgress.connect(self.setProgress)

    self.selfpath = os.path.dirname(os.path.abspath(__file__))
    self.count = 0
    self.name = ''
    self.flag = True
    self.boolSave = False

    face_cascade_file = "haarcascade_frontalface_alt.xml"
    self.face_cascade = cv2.CascadeClassifier(face_cascade_file)

  def finished(self):
    self.flag = False

  # 受付(推論)
  @pyqtSlot()
  def reception(self):
    self.changeLabel.emit("")
    self.datapath = self.selfpath + "/reception/"
    os.makedirs(self.datapath, exist_ok=True)
    self.name = ""
    self.count = 0
    self.takePicture()
    while not os.path.isfile(self.datapath + "0.jpg"):
      QTest.qWait(10)
    name = self.fc.predict(self.datapath + "0.jpg")
    self.changeLabel.emit(name)

  @pyqtSlot(int)
  def setProgress(self, value):
    self.changeProgress.emit(value)

  # 写真を100枚撮影 ➡ データ分割
  @pyqtSlot(str, str)    
  def dataCollection(self, name, mode):
    if mode != "learn":
      self.name = name
      self.datapath = self.selfpath + "/data/" + name + "/"
      os.makedirs(self.datapath, exist_ok=True)
      self.count = 1
      while self.count <= 100:    # 100枚撮影
        print(str(self.count) + " / 100")
        self.takePicture()
        QTest.qWait(100)   # 100msスリープ(UIフリーズなし)
        self.changeProgress.emit(self.count)
      print('撮影終了')

    if mode != "take":
      self.fc.start()

  # 写真を撮影
  def takePicture(self):
    self.boolSave = True

  # カメラ画像をGUIに埋め込む
  def run(self):
    cap = cv2.VideoCapture(self.CAMERA_MODE)
    while self.flag:
      ret, frame = cap.read()
      if ret == False:
        continue
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

      if len(faces) == 1:
        for (x, y, w, h) in faces:
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 1)

        if self.boolSave:
          face = frame[y:y+h, x:x+w]
          path = self.datapath + self.name + str(self.count) + ".jpg"
          self.imwrite(path, face)
          self.boolSave = False
      
      rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      height, width, channels = rgbImage.shape
      bytesPerLine = channels * width
      convertToQtFormat = QImage(rgbImage.data, width, height, bytesPerLine, QImage.Format_RGB888)
      p = convertToQtFormat.scaled(1024, 600, Qt.KeepAspectRatio)
      self.changePixmap.emit(p)
  
  # 画像ファイルを保存
  def imwrite(self, filename, img, params=None):
    try:
      extension = os.path.splitext(filename)[1]
      result, n = cv2.imencode(extension, img, params)

      if result:
        with open(filename, mode='w+b') as f:
          n.tofile(f)
        self.count += 1
        return True
      else:
        return False
    except Exception as e:
      print(e)
      return False
