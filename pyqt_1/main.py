from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QMainWindow, QApplication, QFileDialog,
                             QPushButton, QMessageBox, QLabel)
from PyQt5 import uic
import sys
import cv2

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi('main2.ui', self)

        self.image = None
        self.original_pixmap = None

        # Define the label which will display the image on
        self.label = self.findChild(QLabel, 'label_4')

        self.actionOpen.triggered.connect(self.open_image)
        self.pushButton.clicked.connect(self.convert_2gray)
        self.pushButton_2.clicked.connect(self.restore_original)

        self.pushButton_3.clicked.connect(self.remove_image)

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users\\Mohammad\\Downloads', 'All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg);; Webp Files (*webp)')

        if fname:
            self.image = cv2.imread(fname)
            if self.image is None:
                QMessageBox.warning(self, 'Error', 'Failed to load image: {}'.format(fname))
            else:
                height, width, channel = self.image.shape
                bytes_per_line = width * channel
                q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_BGR888)
                self.original_pixmap = QPixmap.fromImage(q_img)
                self.label.setPixmap(self.original_pixmap)

    def convert_2gray(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            height, width = gray_image.shape
            bytes_per_line = width
            q_img = QImage(gray_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.label.setPixmap(QPixmap.fromImage(q_img))

    def restore_original(self):
        if self.original_pixmap is not None:
            self.label.setPixmap(self.original_pixmap)

    def remove_image(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    UIWindow = UI()
    UIWindow.show()
    sys.exit(app.exec_())
