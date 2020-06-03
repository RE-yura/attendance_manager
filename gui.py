#!/usr/bin/python3
# coding: utf-8
import sys
from PyQt5 import QtTest
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from camera_thread import CameraThread
from face_learning import FaceLearner

# === メインウィンドウ =================================
class MainWindow(QMainWindow):
  triggerDataCollection = pyqtSignal(str, str)
  triggerReception = pyqtSignal()

  def __init__(self):
    super().__init__()
    exitAction = QAction(QIcon.fromTheme('application-exit'), '&Exit', self)
    exitAction.setShortcut('Ctrl+Q')
    exitAction.setStatusTip('Exit application')    # カーソルを乗せるとステータスバーへ表示
    exitAction.triggered.connect(qApp.quit)

    # === ツールバー ========================
    self.toolbar = self.addToolBar('toolbar')
    self.toolbar.setIconSize(QSize(50,50))
    self.toolbar.addAction(exitAction)
    
    self.initUI()

  @pyqtSlot(str)
  def setText(self, text):
    if text == "error":
      self.result_label.setText("顔の登録からスタートして下さい．")
    elif text != "":
      self.result_label.setText('あなたは <span style="color: red">' + text + '</span> ですね?')
    else:
      self.result_label.setText("顔がカメラに上手く映っていません．")

  @pyqtSlot(QImage)
  def setImage(self, image):
    self.monitor.setPixmap(QPixmap.fromImage(image))

  # プログレスバー更新
  @pyqtSlot(int)
  def onProgressChanged(self, value):
    self.progress.setValue(value)

  # 受付ボタン
  def reception(self):
    self.triggerReception.emit()

  # 撮影/学習ボタン
  def dataCollection(self, mode):
    if mode == "learn":
      self.triggerDataCollection.emit(self.name_input.text(), mode)
    else:
      name = self.name_input.text()
      if name == "":
        QMessageBox.warning(self, "Warning", "名前を入力して下さい．")
      else:
        self.triggerDataCollection.emit(self.name_input.text(), mode)

  @pyqtSlot()
  def showPopup(self):
    QMessageBox.information(self, "Message", "始めに顔の登録をして下さい．")
    # msg_box = QMessageBox()
    # msg_box.setAttribute(Qt.WA_DeleteOnClose, True)
    # msg_box.setWindowTitle("Message")
    # msg_box.setStyleSheet("width: 400px;")
    # msg_box.setText('顔の登録をして下さい．')
    # msg_box.setIcon(QMessageBox.Information)
    # restart_btn = msg_box.addButton("OK", QMessageBox.ActionRole)
    # msg_box.setDefaultButton(restart_btn)
    # msg_box.exec_()
  
  def initUI(self):
    # === 中央ウィジェットの生成 =======
    centralWidget = QWidget()
    self.setCentralWidget(centralWidget)
        
    # === メインレイアウトの配置 =======
    main_vlayout = QVBoxLayout()
    centralWidget.setLayout(main_vlayout)

    # === メインレイアウトの中身 ======
    upper_hlayout = QHBoxLayout()
    lower_hlayout = QHBoxLayout()
    main_vlayout.addLayout(upper_hlayout)
    main_vlayout.addLayout(lower_hlayout)

    # === upper_hlayoutの中身 ======
    self.monitor = QLabel()
    # self.monitor.resize(640, 480)
    upper_hlayout.addWidget(self.monitor)

    self.result_label = QLabel()
    self._font = QFont()
    self._font.setPointSize(16)
    self._font.setBold(True)
    self.result_label.setFont(self._font)
    upper_hlayout.addWidget(self.result_label)
 
    # === lower_hlayoutの中身 ======
    tab = QTabWidget()
    lower_hlayout.addWidget(tab)
    
    # ==== タブ1 ===================
    tab1 = QWidget()
    tab.addTab(tab1, "受付")

    tab1_layout = QHBoxLayout()
    tab1.setContentsMargins(100,100,100,100)
    tab1.setLayout(tab1_layout)

    reception_btn = QPushButton('受付')
    reception_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    reception_btn.clicked.connect(self.reception)
    tab1_layout.addWidget(reception_btn)

    # ==== タブ2 ===================
    tab2 = QWidget()
    tab.addTab(tab2, "登録")

    tab2_layout = QHBoxLayout()
    tab2.setContentsMargins(0,0,0,0)
    tab2.setLayout(tab2_layout)

    vlayout = QVBoxLayout()
    vlayout.setAlignment(Qt.AlignCenter)
    tab2_layout.addLayout(vlayout)


    self.progress = QProgressBar()
    self.progress.setMaximum(100)
    self.progress.setValue(0)
    vlayout.addWidget(self.progress)

    form_layout = QFormLayout()
    form_layout.setContentsMargins(0,200,0,0)
    vlayout.addLayout(form_layout)

    name_label = QLabel("名前を入力して下さい")
    self.name_input = QLineEdit()
    self.name_input.setStyleSheet("QLineEdit:focus {background: #e9f7f5}")
    form_layout.addRow(name_label, self.name_input)

    tab2_btns_layout = QVBoxLayout()
    tab2_layout.addLayout(tab2_btns_layout)

    take_learn_btn = QPushButton('撮影 and 学習')
    take_learn_btn.clicked.connect(lambda: self.dataCollection("both"))
    take_learn_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    tab2_btns_layout.addWidget(take_learn_btn)

    take_btn = QPushButton('撮影')
    take_btn.clicked.connect(lambda: self.dataCollection("take"))
    take_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    tab2_btns_layout.addWidget(take_btn)

    learn_btn = QPushButton('学習')
    learn_btn.clicked.connect(lambda: self.dataCollection("learn"))
    learn_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    tab2_btns_layout.addWidget(learn_btn)
 
    self.show()

# === main process ================================
def main():
  app = QApplication(sys.argv)
  desktop = app.desktop()
  height = desktop.height()
  width = desktop.width()

  window = MainWindow()
  window.setGeometry(200, 200, width*0.8, height*0.8)
  lthread = FaceLearner()
  cthread = CameraThread(lthread)

  window.triggerDataCollection.connect(cthread.dataCollection)
  window.triggerReception.connect(cthread.reception)
  cthread.changePixmap.connect(window.setImage)
  cthread.changeLabel.connect(window.setText)
  cthread.changeProgress.connect(window.onProgressChanged)
  lthread.changeProgress.connect(cthread.setProgress)
  lthread.triggerPopup.connect(window.showPopup)
  lthread.loadData()

  window.show()
  cthread.start()

  app.exec_()

  cthread.finished()
  cthread.quit()
  cthread.wait()
  sys.exit(0)

    
if __name__ == '__main__':
  main()
 