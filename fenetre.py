from PyQt5 import QtWidgets, QtGui, QtCore
import sys
from PyQt5.QtGui import QPixmap, QBrush, QPainter, QImage, QPainterPath, QPen
from PyQt5.QtCore import QRect, QPoint, QSize
from PyQt5.QtWidgets import QMessageBox, QApplication, QWidget
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QLabel, QStyleFactory
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class Drawer(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        #self.setAttribute(Qt.WA_StaticContents)
        self.myPenWidth = 25
        self.myPenColor = QtCore.Qt.black
        self.image = QImage(336, 336, QImage.Format_RGB32)
        self.path = QPainterPath()
        self.clearImage()

    def clearImage(self):
        self.path = QPainterPath()
        self.image.fill(QtCore.Qt.white)
        self.update()

    def saveImage(self, fileName, fileFormat):
        self.image.save(fileName, fileFormat)
        img = cv2.imread('image.jpg',0)
        img = 1-img/255
        newimg = cv2.resize(img, (28,28))
        gblimg = cv2.GaussianBlur(newimg, (3, 3), 1)
        X=gblimg.reshape(1,28,28,1)
        proba = model.predict(X)
        if np.max(proba) >= 0.6: 
            result = np.argmax(proba)
            print("c'est un "+ str(result))
            self.msg = QMessageBox()
            self.msg.setWindowTitle("C'est presque sûr...")
            self.msg.setStyle(QStyleFactory.create("Fusion"))
            self.msg.setStyleSheet("background: #77bb66;")
            self.msg.setText(f"Ce chiffre est un {result} et c'est sûr à {round(np.max(proba)*100,1)} %")
            x = self.msg.exec_()
            self.clearImage()
        else :
            self.msg2 = QMessageBox()
            self.msg2.setWindowTitle("Pas terrible...")
            self.msg2.setStyle(QStyleFactory.create("Fusion"))
            self.msg2.setStyleSheet("background: #cc6666;")
            self.msg2.setText(f"Là, vraiment, je vois pas... Réessaye !")
            x = self.msg2.exec_()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image, self.rect())

    def mousePressEvent(self, event):
        self.path.moveTo(event.pos())

    def mouseMoveEvent(self, event):
        self.path.lineTo(event.pos())
        p = QPainter(self.image)
        p.setPen(QPen(self.myPenColor,
                      self.myPenWidth, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                      QtCore.Qt.RoundJoin))
        p.drawPath(self.path)
        p.end()
        self.update()

    def sizeHint(self):
        return QSize(300, 300)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    w = QWidget()
    w.setWindowTitle("TheUltimatePredictor")
    w.setStyleSheet("background: #226611; color: #FFFFFF;") 
    btnSave = QPushButton("Identifier")
    btnClear = QPushButton("Effacer")
    drawer = Drawer()

    model = tf.keras.models.load_model('mnist_model')

    w.setLayout(QVBoxLayout())
    w.layout().addWidget(drawer)
    w.layout().addWidget(btnSave)
    w.layout().addWidget(btnClear)

    btnSave.clicked.connect(lambda: drawer.saveImage("image.jpg", "JPG"))
    btnClear.clicked.connect(drawer.clearImage)

    w.show()
    sys.exit(app.exec_())