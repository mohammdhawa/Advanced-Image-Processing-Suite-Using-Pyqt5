from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QMainWindow, QApplication, QFileDialog,
                             QPushButton, QMessageBox, QLabel,
                             QWidget, QLineEdit, QSlider, QAction)
from PyQt5 import uic
import sys
import cv2
import numpy as np


class UI(QMainWindow):
    def _init_(self):
        super(UI, self)._init_()
        # load the file which created from QT Designer
        uic.loadUi('beyan.ui', self)

        self.image = None
        self.original_image = None  # Store the original image data
        self.original_pixmap = None

        # Define the labels which will display the image on
        self.label = self.findChild(QLabel, 'label_4')
        self.label.setAlignment(Qt.AlignCenter)  # Align label content to center
        self.width_label = self.findChild(QLabel, 'label_8')
        self.height_label = self.findChild(QLabel, 'label_7')

        # Define image QMenu Actions
        self.image_upload = self.findChild(QAction, 'actionUpload')
        self.image_save = self.findChild(QAction, 'actionSave')
        self.image_undo = self.findChild(QAction, 'actionUndo')
        self.image_remove = self.findChild(QAction, 'actionRemove')

        # Define Filters
        self.image_graysacle = self.findChild(QAction, 'actionGrayScale')
        self.image_edge_detection = self.findChild(QAction, 'actionEdge_Detection')
        self.image_dilation = self.findChild(QAction, 'actionDilation')
        self.image_erosion = self.findChild(QAction, 'actionErosion')

        # Define line edits (width & height)
        self.width_line_edit = self.findChild(QLineEdit, 'lineEdit_4')
        self.height_line_edit = self.findChild(QLineEdit, 'lineEdit_3')

        # Define Sliders
        self.brightness_slider = self.findChild(QSlider, 'horizontalSlider')
        self.contrast_slider = self.findChild(QSlider, 'horizontalSlider_2')
        self.blur_slider = self.findChild(QSlider, 'horizontalSlider_3')

        # Methods
        self.image_upload.triggered.connect(self.open_image)
        self.image_save.triggered.connect(self.save_image)
        self.image_undo.triggered.connect(self.undo)
        self.image_remove.triggered.connect(self.remove_image)
        self.image_graysacle.triggered.connect(self.convert_2gray)
        self.image_edge_detection.triggered.connect(self.edge_detection)
        self.image_dilation.triggered.connect(self.dilation)
        self.image_erosion.triggered.connect(self.erosion)
        self.brightness_slider.valueChanged.connect(self.change_brightness)
        self.contrast_slider.valueChanged.connect(self.change_contrast)
        self.blur_slider.valueChanged.connect(self.change_blur)

        # just for testing
        # self.erosion_button.clicked.connect(self.hello_world)

    # Just for testing
    def hello_world(self):
        QMessageBox.about(self, "Title", "Message")

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users\\Mohammad\\Downloads', 'All Files ();;PNG Files (.png);;Jpg Files (*.jpg);; Webp Files (*webp)')

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

    def convert_2gray(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            height, width = gray_image.shape
            bytes_per_line = width
            q_img = QImage(gray_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                                        Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))
            self.image = gray_image

    def undo(self):
        if self.original_image is not None:
            bytes_per_line = self.original_image.shape[1] * self.original_image.shape[2]
            q_img = QImage(self.original_image.data, self.original_image.shape[1], self.original_image.shape[0], bytes_per_line, QImage.Format_BGR888)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))
            self.image = self.original_image.copy()  # Restore the original image data

    def enable_buttons(self):
        self.width_line_edit.setEnabled(True)
        self.height_line_edit.setEnabled(True)
        self.width_label.setEnabled(True)
        self.height_label.setEnabled(True)
        self.brightness_slider.setEnabled(True)
        self.contrast_slider.setEnabled(True)
        self.blur_slider.setEnabled(True)

    def disable_buttons(self):
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

    def dilation(self):
        # Convert the image to grayscale if it's not already in grayscale
        if len(self.image.shape) > 2:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image

        # Define a structuring element (kernel)
        kernel = np.ones((5, 5), np.uint8)

        # Apply dilation
        dilated_image = cv2.dilate(gray_image, kernel, iterations=1)

        # Display the cropped image
        height, width = dilated_image.shape
        bytes_per_line = width
        q_img = QImage(dilated_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation)
        self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

        self.image = dilated_image

    def erosion(self):
        # Convert the image to grayscale if it's not already in grayscale
        if len(self.image.shape) > 2:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image

        # Define a structuring element (kernel)
        kernel = np.ones((5, 5), np.uint8)

        # Apply dilation
        erosion_image = cv2.erode(gray_image, kernel, iterations=1)

        # Display the cropped image
        height, width = erosion_image.shape
        bytes_per_line = width
        q_img = QImage(erosion_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation)
        self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

        self.image = erosion_image

    def save_image(self):
        if self.image is not None:
            # Get the file path to save the image
            fname, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'PNG Files (.png);;JPG Files (.jpg)')
            if fname:
                # Check file extension and add if not present
                if not fname.lower().endswith(('.png', '.jpg', '.webp')):
                    fname += '.png'  # Default to PNG format if extension is missing

                # Save the image
                cv2.imwrite(fname, self.image)
                QMessageBox.information(self, 'Saved', 'Image saved successfully at: {}'.format(fname))

    def change_brightness(self, value):
        if self.image is not None:
            brightness = value - 50  # Translate the slider value to a brightness level (-50 to 50)
            adjusted_image = cv2.convertScaleAbs(self.image, alpha=1, beta=brightness)

            # Display the adjusted image
            height, width, channel = adjusted_image.shape
            bytes_per_line = width * channel
            q_img = QImage(adjusted_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

            # self.image = adjusted_image

    def change_brightness2(self, value):
        if self.image is not None:
            # Normalize the slider value to adjust brightness in the range of -50 to 50
            brightness = value - 50

            # Adjust image brightness using cv2.convertScaleAbs
            adjusted_image = cv2.convertScaleAbs(self.image, beta=brightness)

            # Convert OpenCV image to QImage
            height, width, channel = adjusted_image.shape
            bytes_per_line = channel * width
            q_img = QImage(adjusted_image.data, width, height, bytes_per_line, QImage.Format_BGR888)

            # Convert QImage to QPixmap and display on the label
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

            self.image = adjusted_image

            # Update the image attribute to the adjusted image
            # self.image = adjusted_image

    def change_contrast(self, value):
        if self.image is not None:
            # Translate slider value to a contrast level (-1.0 to 1.0)
            contrast = (value / 50.0) * 2 - 1.0

            # Adjusting the contrast within the range (-1.0 to 1.0)
            if contrast > 0:
                contrast = 1 + contrast
            else:
                contrast = 1 / (1 - contrast)

            adjusted_image = cv2.convertScaleAbs(self.image, alpha=contrast, beta=0)

            # Display the adjusted image
            height, width, channel = adjusted_image.shape
            bytes_per_line = width * channel
            q_img = QImage(adjusted_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

            # self.image = adjusted_image

    def change_blur(self, value):
        if self.image is not None:
            # Define maximum blur radius
            max_blur_radius = 50

            # Translate slider value to blur radius (0 to max_blur_radius)
            blur_radius = max(0, min(abs(value), max_blur_radius))

            # Apply Gaussian blur with the calculated radius
            adjusted_image = cv2.GaussianBlur(self.image, (2 * blur_radius + 1, 2 * blur_radius + 1), 0)

            # Display the adjusted image (same logic as change_contrast)
            height, width, channel = adjusted_image.shape
            bytes_per_line = width * channel
            q_img = QImage(adjusted_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

            # self.image = adjusted_image


if __name__ == "__main__":
    app = QApplication(sys.argv)
    UIWindow = UI()
    UIWindow.show()
    sys.exit(app.exec_())