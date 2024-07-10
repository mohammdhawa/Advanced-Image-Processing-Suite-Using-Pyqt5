from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QMainWindow, QApplication, QFileDialog,
                             QPushButton, QMessageBox, QLabel,
                             QWidget, QLineEdit, QSlider)
from PyQt5 import uic
import sys
import cv2
import numpy as np
from PIL import Image, ImageEnhance


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        # load the file which created from QT Designer
        uic.loadUi('main.ui', self)

        self.image = None
        self.original_image = None  # Store the original image data
        self.original_pixmap = None

        # Define the labels which will display the image on
        self.label = self.findChild(QLabel, 'label_4')
        self.label.setAlignment(Qt.AlignCenter)  # Align label content to center
        self.width_label = self.findChild(QLabel, 'label_8')
        self.height_label = self.findChild(QLabel, 'label_7')

        # Define buttons
        self.load_image_button = self.findChild(QPushButton, 'pushButton_7')
        self.save_image_button = self.findChild(QPushButton, 'pushButton')
        self.undo_button = self.findChild(QPushButton, 'pushButton_5')
        self.remove_button = self.findChild(QPushButton, 'pushButton_8')
        self.cvt2gray_button = self.findChild(QPushButton, 'pushButton_9')
        self.edge_detection_button = self.findChild(QPushButton, 'pushButton_21')


        # Define line edits (width & height)
        self.width_line_edit = self.findChild(QLineEdit, 'lineEdit_4')
        self.height_line_edit = self.findChild(QLineEdit, 'lineEdit_3')

        # Methods
        self.load_image_button.clicked.connect(self.open_image)
        self.cvt2gray_button.clicked.connect(self.convert_2gray)
        self.undo_button.clicked.connect(self.undo)
        self.remove_button.clicked.connect(self.remove_image)
        self.edge_detection_button.clicked.connect(self.edge_detection)
        self.dilation_button.clicked.connect(self.dilation)
        self.erosion_button.clicked.connect(self.erosion)
        self.save_image_button.clicked.connect(self.save_image)
        self.brightness_slider.valueChanged.connect(self.change_brightness)
        self.contrast_slider.valueChanged.connect(self.change_contrast)
        self.blur_slider.valueChanged.connect(self.change_blur)

        # just for testing
        # self.erosion_button.clicked.connect(self.hello_world)

    # Just for testing
    def hello_world(self):
        QMessageBox.about(self, "Title", "Message")

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users\\Mohammad\\Downloads',
                                               'All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg);; Webp Files (*webp)')

        if fname:
            self.image = cv2.imread(fname)
            self.original_image = self.image.copy()
            if self.image is None:
                QMessageBox.warning(self, 'Error', 'Failed to load image: {}'.format(fname))
            else:
                height, width, channel = self.image.shape
                bytes_per_line = width * channel
                q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_BGR888)
                self.original_pixmap = QPixmap.fromImage(q_img)
                # Scale the pixmap to fit the label while maintaining aspect ratio
                scaled_pixmap = self.original_pixmap.scaled(self.label.size(), Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation)
                self.label.setPixmap(scaled_pixmap)
                # enabling buttons after uploading an image
                self.enable_buttons()
                # setting width & height
                self.get_size()

    def enable_buttons(self):
        self.save_image_button.setEnabled(True)
        self.undo_button.setEnabled(True)
        self.remove_button.setEnabled(True)
        self.cvt2gray_button.setEnabled(True)
        self.edge_detection_button.setEnabled(True)
        self.dilation_button.setEnabled(True)
        self.erosion_button.setEnabled(True)
        self.width_line_edit.setEnabled(True)
        self.height_line_edit.setEnabled(True)
        self.width_label.setEnabled(True)
        self.height_label.setEnabled(True)
        self.brightness_slider.setEnabled(True)
        self.contrast_slider.setEnabled(True)
        self.blur_slider.setEnabled(True)

    def disable_buttons(self):
        self.save_image_button.setEnabled(False)
        self.undo_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.cvt2gray_button.setEnabled(False)
        self.edge_detection_button.setEnabled(False)
        self.dilation_button.setEnabled(False)
        self.erosion_button.setEnabled(False)
        self.width_line_edit.setEnabled(False)
        self.height_line_edit.setEnabled(False)
        self.width_label.setEnabled(False)
        self.height_label.setEnabled(False)
        self.brightness_slider.setEnabled(False)
        self.contrast_slider.setEnabled(False)
        self.blur_slider.setEnabled(False)

    def get_size(self):
        if self.image is not None:
            self.width, self.height, _ = self.image.shape
            self.width_line_edit.setText(str(self.width))
            self.height_line_edit.setText(str(self.height))

    def remove_image(self):
        self.label.setPixmap(QPixmap())
        self.disable_buttons()
        self.width_line_edit.clear()
        self.height_line_edit.clear()

    def undo(self):
        if self.original_image is not None:
            bytes_per_line = self.original_image.shape[1] * self.original_image.shape[2]
            q_img = QImage(self.original_image.data, self.original_image.shape[1], self.original_image.shape[0],
                           bytes_per_line, QImage.Format_BGR888)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))
            self.image = self.original_image.copy()  # Restore the original image data

    def convert_2gray(self):
        if self.image is not None:
            if len(self.image.shape) == 3:  # Check if it's a BGR image
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = self.image

            height, width = gray_image.shape
            bytes_per_line = width
            q_img = QImage(gray_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))
            self.image = gray_image

    def edge_detection(self):
        # Convert the image to grayscale if it's not already in grayscale
        if len(self.image.shape) > 2:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image

        # Perform Canny edge detection
        edges = cv2.Canny(gray_image, 100, 200)  # Adjust the thresholds as needed

        # Display the edges image
        height, width = edges.shape
        bytes_per_line = width
        q_img = QImage(edges.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation)
        self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

        self.image = edges

    def save_image(self):
        if self.image is not None:
            # Get the file path to save the image
            fname, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'PNG Files (*.png);;JPG Files (*.jpg)')
            if fname:
                # Check file extension and add if not present
                if not fname.lower().endswith(('.png', '.jpg', '.webp')):
                    fname += '.png'  # Default to PNG format if extension is missing

                # Save the image
                cv2.imwrite(fname, self.image)
                QMessageBox.information(self, 'Saved', 'Image saved successfully at: {}'.format(fname))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    UIWindow = UI()
    UIWindow.show()
    sys.exit(app.exec_())
