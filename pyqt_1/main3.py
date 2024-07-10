from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QMainWindow, QApplication, QFileDialog,
                             QPushButton, QMessageBox, QLabel,
                             QWidget, QLineEdit, QSlider)
from PyQt5 import uic
import sys
import cv2
import numpy as np


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        # load the file which created from QT Designer
        uic.loadUi('main.ui', self)

        self.image = None
        self.original_image = None  # Store the original image data
        self.original_pixmap = None
        self.image_stack = []
        self.redo_stack = []  # Initialize redo stack

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
        self.dilation_button = self.findChild(QPushButton, 'pushButton_28')
        self.erosion_button = self.findChild(QPushButton, 'pushButton_29')
        self.opening_button = self.findChild(QPushButton, 'pushButton_35')
        self.closing_button = self.findChild(QPushButton, 'pushButton_36')
        self.redo_button = self.findChild(QPushButton, 'pushButton_2')
        self.histogram_equalization_button = self.findChild(QPushButton, 'pushButton_30')
        self.invert_color_button = self.findChild(QPushButton, 'pushButton_13')
        self.simple_blur_button = self.findChild(QPushButton, 'pushButton_10')
        self.gaussian_blur_button = self.findChild(QPushButton, 'pushButton_22')
        self.median_blur_button = self.findChild(QPushButton, 'pushButton_31')
        self.bilatral_blur_button = self.findChild(QPushButton, 'pushButton_32')
        self.sharpend_button = self.findChild(QPushButton, 'pushButton_33')
        self.box_filter_button = self.findChild(QPushButton, 'pushButton_34')
        self.theme_button = self.findChild(QPushButton, 'pushButton_3')
        self.theme_button2 = self.findChild(QPushButton, 'pushButton_4')
        self.theme_button3 = self.findChild(QPushButton, 'pushButton_6')
        self.theme_button4 = self.findChild(QPushButton, 'pushButton_11')

        # Define line edits (width & height)
        self.width_line_edit = self.findChild(QLineEdit, 'lineEdit_4')
        self.height_line_edit = self.findChild(QLineEdit, 'lineEdit_3')

        # Define Sliders
        self.brightness_slider = self.findChild(QSlider, 'horizontalSlider')
        self.contrast_slider = self.findChild(QSlider, 'horizontalSlider_2')

        # Methods
        self.load_image_button.clicked.connect(self.open_image)
        self.cvt2gray_button.clicked.connect(self.convert_2gray)
        self.undo_button.clicked.connect(self.undo)
        self.remove_button.clicked.connect(self.remove_image)
        self.edge_detection_button.clicked.connect(self.edge_detection)
        self.redo_button.clicked.connect(self.redo)
        self.dilation_button.clicked.connect(self.dilation)
        self.erosion_button.clicked.connect(self.erosion)
        self.opening_button.clicked.connect(self.opening)
        self.closing_button.clicked.connect(self.closing)
        self.histogram_equalization_button.clicked.connect(self.histogram_equalization)
        self.invert_color_button.clicked.connect(self.invert_colors)
        self.save_image_button.clicked.connect(self.save_image)
        self.brightness_slider.valueChanged.connect(self.change_brightness)
        self.contrast_slider.valueChanged.connect(self.change_contrast)

        self.simple_blur_button.clicked.connect(self.simple_blur)
        self.gaussian_blur_button.clicked.connect(self.gaussian_blur)
        self.median_blur_button.clicked.connect(self.median_blur)
        self.sharpend_button.clicked.connect(self.sharpen)
        self.box_filter_button.clicked.connect(self.box_filter)
        self.theme_button.clicked.connect(self.Apply_QToolery_Style)
        self.theme_button2.clicked.connect(self.Apply_QSynet_Style)
        self.theme_button3.clicked.connect(self.Apply_QDarkBlue_Style)
        self.theme_button4.clicked.connect(self.Apply_QDark_Style)



        # just for testing
        # self.erosion_button.clicked.connect(self.hello_world)

    # Just for testing
    def hello_world(self):
        QMessageBox.about(self, "Title", "Message")

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
        self.histogram_equalization_button.setEnabled(True)
        self.opening_button.setEnabled(True)
        self.closing_button.setEnabled(True)
        self.invert_color_button.setEnabled(True)
        self.simple_blur_button.setEnabled(True)
        self.gaussian_blur_button.setEnabled(True)
        self.median_blur_button.setEnabled(True)
        self.bilatral_blur_button.setEnabled(True)
        self.sharpend_button.setEnabled(True)
        self.box_filter_button.setEnabled(True)

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
        self.redo_button.setEnabled(False)
        self.histogram_equalization_button.setEnabled(False)
        self.opening_button.setEnabled(False)
        self.closing_button.setEnabled(False)
        self.invert_color_button.setEnabled(False)
        self.simple_blur_button.setEnabled(False)
        self.gaussian_blur_button.setEnabled(False)
        self.median_blur_button.setEnabled(False)
        self.bilatral_blur_button.setEnabled(False)
        self.sharpend_button.setEnabled(False)
        self.box_filter_button.setEnabled(False)

    def get_size(self):
        if len(self.image_stack):
            image = self.image_stack[-1]
            if image is not None:
                self.width, self.height, _ = image.shape
                self.width_line_edit.setText(f"{str(self.width)}px")
                self.height_line_edit.setText(f"{str(self.height)}px")

    def remove_image(self):
        self.disable_buttons()
        self.width_line_edit.clear()
        self.height_line_edit.clear()
        self.brightness_slider.setValue(0)  # Reset brightness slider
        self.contrast_slider.setValue(0)
        self.image_stack.clear()  # Clear the image stack to remove all adjustments
        self.redo_stack.clear()  # Clear redo stack as well
        self.original_image = None  # Reset the original image
        self.original_pixmap = None
        self.label.clear()

    def save_image(self):
        if len(self.image_stack):
            image = self.image_stack[-1]
            image = self.change_brightness2(image, self.brightness_slider.value())
            image = self.change_contrast2(image, self.contrast_slider.value())


            if image is not None:
                # Get the file path to save the image
                fname, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'PNG Files (*.png);;JPG Files (*.jpg)')
                if fname:
                    # Check file extension and add if not present
                    if not fname.lower().endswith(('.png', '.jpg', '.webp')):
                        fname += '.png'  # Default to PNG format if extension is missing

                    # Save the image
                    cv2.imwrite(fname, image)
                    QMessageBox.information(self, 'Saved', 'Image saved successfully at: {}'.format(fname))

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users\\Mohammad\\Downloads',
                                               'All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg);; Webp Files (*webp)')

        if fname:
            image = cv2.imread(fname)
            self.original_image = image.copy()
            self.image_stack.clear()
            self.image_stack.append(image.copy())
            if image is None:
                QMessageBox.warning(self, 'Error', 'Failed to load image: {}'.format(fname))
            else:
                height, width, channel = image.shape
                bytes_per_line = width * channel
                q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
                self.original_pixmap = QPixmap.fromImage(q_img)
                # Scale the pixmap to fit the label while maintaining aspect ratio
                scaled_pixmap = self.original_pixmap.scaled(self.label.size(), Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation)
                self.label.setPixmap(scaled_pixmap)
                # enabling buttons after uploading an image
                self.enable_buttons()
                # setting width & height
                self.get_size()

    def undo(self):
        if len(self.image_stack) > 1:
            self.redo_stack.append(self.image_stack.pop())
            if self.redo_stack:
                self.redo_button.setEnabled(True)
            image = self.image_stack[-1]
        elif len(self.image_stack) == 1:
            image = self.original_image
            self.image_stack.clear()
            self.image_stack.append(image)
        print("len of st: ", len(self.image_stack))
        if len(image.shape) == 3:  # check if it's a BGR image
            bytes_per_line = image.shape[1] * image.shape[2]
            q_img = QImage(image.data, image.shape[1], image.shape[0],
                           bytes_per_line, QImage.Format_BGR888)
        else:  # if it's grayscale
            bytes_per_line = image.shape[1]
            q_img = QImage(image.data, image.shape[1], image.shape[0],
                           bytes_per_line, QImage.Format_Grayscale8)

        scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation)
        self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

    def redo(self):
        if self.redo_stack:
            self.image_stack.append(self.redo_stack.pop())  # Move the redone image to image stack
            image = self.image_stack[-1]
            if len(image.shape) == 3:  # check if it's a BGR image
                bytes_per_line = image.shape[1] * image.shape[2]
                q_img = QImage(image.data, image.shape[1], image.shape[0],
                               bytes_per_line, QImage.Format_BGR888)
            else:  # if it's grayscale
                bytes_per_line = image.shape[1]
                q_img = QImage(image.data, image.shape[1], image.shape[0],
                               bytes_per_line, QImage.Format_Grayscale8)

            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

        if len(self.redo_stack) == 0:
            self.redo_button.setEnabled(False)

    def convert_2gray(self):
        if self.image_stack:
            image = self.image_stack[-1]
            if len(image.shape) == 3:  # Check if it's a BGR image
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            height, width = gray_image.shape
            bytes_per_line = width
            q_img = QImage(gray_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))
            # self.image = gray_image
            self.image_stack.append(gray_image)
            print(len(self.image_stack))

    def edge_detection(self):
        if self.image_stack:
            image = self.image_stack[-1]
            # Convert the image to grayscale if it's not already in grayscale
            if len(image.shape) > 2:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Perform Canny edge detection
            edges = cv2.Canny(gray_image, 100, 200)  # Adjust the thresholds as needed

            # Display the edges image
            height, width = edges.shape
            bytes_per_line = width
            q_img = QImage(edges.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

            # self.image = edges
            self.image_stack.append(edges)
            print(len(self.image_stack))

    def dilation(self):
        if len(self.image_stack):
            image = self.image_stack[-1]
            # Convert the image to grayscale if it's not already in grayscale
            if len(image.shape) > 2:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

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

            # self.image = edges
            self.image_stack.append(dilated_image)
            print(len(self.image_stack))

    def erosion(self):
        if len(self.image_stack):
            image = self.image_stack[-1]
            # Convert the image to grayscale if it's not already in grayscale
            if len(image.shape) > 2:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

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

            # self.image = edges
            self.image_stack.append(erosion_image)
            print(len(self.image_stack))

    def opening(self):
        if len(self.image_stack):
            image = self.image_stack[-1]
            # Convert the image to grayscale if it's not already in grayscale
            if len(image.shape) > 2:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Define a structuring element (kernel)
            kernel = np.ones((5, 5), np.uint8)

            # Apply Opening
            opening_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

            # Display the cropped image
            height, width = opening_image.shape
            bytes_per_line = width
            q_img = QImage(opening_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

            # self.image = edges
            self.image_stack.append(opening_image)
            print(len(self.image_stack))

    def closing(self):
        if len(self.image_stack):
            image = self.image_stack[-1]
            # Convert the image to grayscale if it's not already in grayscale
            if len(image.shape) > 2:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Define a structuring element (kernel)
            kernel = np.ones((5, 5), np.uint8)

            # Apply Closing
            closing_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

            # Display the cropped image
            height, width = closing_image.shape
            bytes_per_line = width
            q_img = QImage(closing_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

            # self.image = edges
            self.image_stack.append(closing_image)
            print(len(self.image_stack))

    def change_brightness(self, value):
        if len(self.image_stack):
            image = self.image_stack[-1]
            if image is not None:
                if value == 0:
                    # If value is 0, keep the original image
                    adjusted_image = image.copy()
                else:
                    brightness = value  # Translate the slider value to a brightness level (-50 to 50)
                    if len(image.shape) == 3:  # Color image
                        adjusted_image = cv2.convertScaleAbs(image, alpha=1, beta=brightness)
                    else:  # Grayscale image
                        adjusted_image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

                # Display the adjusted image
                height, width = adjusted_image.shape[:2]  # Grayscale image has no channel
                if len(adjusted_image.shape) == 2:  # Grayscale image
                    channel = 1
                else:  # Color image
                    channel = adjusted_image.shape[2]
                bytes_per_line = width * channel
                if channel == 1:
                    format = QImage.Format_Grayscale8
                else:
                    format = QImage.Format_BGR888
                q_img = QImage(adjusted_image.data, width, height, bytes_per_line, format)
                scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

    def change_brightness2(self, image, value):
        if image is not None:
            if value == 0:
                # If value is 0, keep the original image
                adjusted_image = image.copy()
            else:
                brightness = value  # Translate the slider value to a brightness level (-50 to 50)
                if len(image.shape) == 3:  # Color image
                    adjusted_image = cv2.convertScaleAbs(image, alpha=1, beta=brightness)
                else:  # Grayscale image
                    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

            return adjusted_image

    def change_contrast(self, value):
        if len(self.image_stack):
            image = self.image_stack[-1]
            if image is not None:
                # Translate slider value to a contrast level (-1.0 to 1.0)
                contrast = (value / 50.0) * 2 - 1.0

                # Adjusting the contrast within the range (-1.0 to 1.0)
                if contrast > 0:
                    contrast = 1 + contrast
                else:
                    contrast = 1 / (1 - contrast)

                if len(image.shape) == 3:
                    # Color image
                    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
                else:
                    # Grayscale image
                    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)

                # Display the adjusted image
                height, width = adjusted_image.shape[:2]
                if len(adjusted_image.shape) == 3:
                    channel = adjusted_image.shape[2]
                    bytes_per_line = width * channel
                    q_img = QImage(adjusted_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
                else:
                    bytes_per_line = width
                    q_img = QImage(adjusted_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

                scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

    def change_contrast2(self, image, value):
        if image is not None:
            # Translate slider value to a contrast level (-1.0 to 1.0)
            contrast = (value / 50.0) * 2 - 1.0

            # Adjusting the contrast within the range (-1.0 to 1.0)
            if contrast > 0:
                contrast = 1 + contrast
            else:
                contrast = 1 / (1 - contrast)

            if len(image.shape) == 3:
                adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
            else:
                adjusted_image = cv2.convertScaleAbs(image, alpha=1, beta=0)  # No contrast adjustment for grayscale

            return adjusted_image

    def histogram_equalization(self):
        if len(self.image_stack):
            image = self.image_stack[-1]
            if len(image.shape) > 2:
                # Convert the image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Apply histogram equalization
            equalized_image = cv2.equalizeHist(gray_image)

            # Display the equalized image
            height, width = equalized_image.shape
            bytes_per_line = width
            q_img = QImage(equalized_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

            # Update the image stack
            self.image_stack.append(equalized_image)

    def invert_colors(self):
        if self.image_stack:
            image = self.image_stack[-1]

            # Invert the image colors
            inverted_image = cv2.bitwise_not(image)

            # Determine the format of the QImage based on the number of channels in the image
            if len(inverted_image.shape) == 2:
                height, width = inverted_image.shape
                bytes_per_line = width
                q_img_format = QImage.Format_Grayscale8
            else:
                height, width, channels = inverted_image.shape
                bytes_per_line = channels * width
                q_img_format = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888

            # Convert the inverted image to QImage
            q_img = QImage(inverted_image.data, width, height, bytes_per_line, q_img_format)
            scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

            # Update the image stack
            self.image_stack.append(inverted_image)
            print(len(self.image_stack))

    def simple_blur(self):
        if len(self.image_stack):
            image = self.image_stack[-1]

            # Apply Simple Blur
            blurred_image = cv2.blur(image, (5, 5))

            # Check if the image is grayscale or color
            if len(blurred_image.shape) == 2:
                height, width = blurred_image.shape
                bytes_per_line = width
                q_img = QImage(blurred_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                height, width, channel = blurred_image.shape
                bytes_per_line = 3 * width
                q_img = QImage(blurred_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Display the blurred image
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(self.label.size(), Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

            # Append the blurred image to the image stack
            self.image_stack.append(blurred_image)
            print(len(self.image_stack))

    def gaussian_blur(self):
        if len(self.image_stack):
            image = self.image_stack[-1]

            # Apply Gaussian Blur
            blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

            # Check if the image is grayscale or color
            if len(blurred_image.shape) == 2:
                height, width = blurred_image.shape
                bytes_per_line = width
                q_img = QImage(blurred_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                height, width, channel = blurred_image.shape
                bytes_per_line = 3 * width
                q_img = QImage(blurred_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Display the blurred image
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(self.label.size(), Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

            # Append the blurred image to the image stack
            self.image_stack.append(blurred_image)
            print(len(self.image_stack))

    def median_blur(self):
        if len(self.image_stack):
            image = self.image_stack[-1]

            # Apply Median Blur
            blurred_image = cv2.medianBlur(image, 5)

            # Check if the image is grayscale or color
            if len(blurred_image.shape) == 2:
                height, width = blurred_image.shape
                bytes_per_line = width
                q_img = QImage(blurred_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                height, width, channel = blurred_image.shape
                bytes_per_line = 3 * width
                q_img = QImage(blurred_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Display the blurred image
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(self.label.size(), Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

            # Append the blurred image to the image stack
            self.image_stack.append(blurred_image)
            print(len(self.image_stack))

    def sharpen(self):
        if len(self.image_stack):
            image = self.image_stack[-1]

            # Apply Sharpening Filter
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            sharpened_image = cv2.filter2D(image, -1, kernel)

            # Check if the image is grayscale or color
            if len(sharpened_image.shape) == 2:
                height, width = sharpened_image.shape
                bytes_per_line = width
                q_img = QImage(sharpened_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                height, width, channel = sharpened_image.shape
                bytes_per_line = 3 * width
                q_img = QImage(sharpened_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Display the sharpened image
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(self.label.size(), Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

            # Append the sharpened image to the image stack
            self.image_stack.append(sharpened_image)
            print(len(self.image_stack))

    def box_filter(self):
        if len(self.image_stack):
            image = self.image_stack[-1]

            # Apply Box Filter
            box_filtered_image = cv2.boxFilter(image, -1, (5, 5), normalize=True)

            # Check if the image is grayscale or color
            if len(box_filtered_image.shape) == 2:
                height, width = box_filtered_image.shape
                bytes_per_line = width
                q_img = QImage(box_filtered_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                height, width, channel = box_filtered_image.shape
                bytes_per_line = 3 * width
                q_img = QImage(box_filtered_image.data, width, height, bytes_per_line,
                               QImage.Format_RGB888).rgbSwapped()

            # Display the box filtered image
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(self.label.size(), Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

            # Append the box filtered image to the image stack
            self.image_stack.append(box_filtered_image)
            print(len(self.image_stack))

    ###### App Themes ####

    def Apply_QDark_Style(self):
        style = open('themes/dark.css', 'r')
        style = style.read()
        self.setStyleSheet(style)

    def Apply_QDarkBlue_Style(self):
        style = open('themes/darkblu.css', 'r')
        style = style.read()
        self.setStyleSheet(style)

    def Apply_QDarkOrange_Style(self):
        style = open('themes/darkorange.css', 'r')
        style = style.read()
        self.setStyleSheet(style)

    def Apply_QToolery_Style(self):
        style = open('themes/toolery.css', 'r')
        style = style.read()
        self.setStyleSheet(style)

    def Apply_QSybot_Style(self):
        style = open('themes/sybot.css', 'r')
        style = style.read()
        self.setStyleSheet(style)

    def Apply_QPicpax_Style(self):
        style = open('themes/picpax.css', 'r')
        style = style.read()
        self.setStyleSheet(style)

    def Apply_QSynet_Style(self):
        style = open('themes/synet.css', 'r')
        style = style.read()
        self.setStyleSheet(style)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    UIWindow = UI()
    UIWindow.show()
    sys.exit(app.exec_())
