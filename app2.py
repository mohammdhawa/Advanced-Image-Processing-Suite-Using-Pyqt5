from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QMainWindow, QApplication, QFileDialog,
                             QPushButton, QMessageBox, QLabel,
                             QWidget, QLineEdit, QSlider, QVBoxLayout,
                             QScrollArea, QDialog)
from PyQt5 import uic
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


class ThresholdDialog(QDialog):
    def __init__(self, parent=None):
        super(ThresholdDialog, self).__init__(parent)
        self.setWindowTitle("Edge Detection Thresholds")

        self.layout = QVBoxLayout()

        self.label1 = QLabel("Lower Threshold:")
        self.layout.addWidget(self.label1)
        self.lower_threshold_input = QLineEdit(self)
        self.layout.addWidget(self.lower_threshold_input)

        self.label2 = QLabel("Upper Threshold:")
        self.layout.addWidget(self.label2)
        self.upper_threshold_input = QLineEdit(self)
        self.layout.addWidget(self.upper_threshold_input)

        self.button = QPushButton("Apply", self)
        self.button.clicked.connect(self.apply_thresholds)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

        self.lower_threshold = None
        self.upper_threshold = None

    def apply_thresholds(self):
        try:
            self.lower_threshold = int(self.lower_threshold_input.text())
            self.upper_threshold = int(self.upper_threshold_input.text())
            self.accept()
        except ValueError:
            # Handle the case where the input is not a valid integer
            print("Please enter valid integer thresholds.")


class SobelDialog(QDialog):
    def __init__(self, parent=None):
        super(SobelDialog, self).__init__(parent)
        self.setWindowTitle("Sobel Detection Kernel Size")

        self.layout = QVBoxLayout()

        self.label = QLabel("Kernel Size (must be an odd integer):")
        self.layout.addWidget(self.label)
        self.kernel_size_input = QLineEdit(self)
        self.layout.addWidget(self.kernel_size_input)

        self.button = QPushButton("Apply", self)
        self.button.clicked.connect(self.apply_kernel_size)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

        self.kernel_size = None

    def apply_kernel_size(self):
        try:
            kernel_size = int(self.kernel_size_input.text())
            if kernel_size % 2 == 1 and kernel_size > 0:
                self.kernel_size = kernel_size
                self.accept()
            else:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter a valid positive odd integer.")


class ResizeDialog(QDialog):
    def __init__(self, parent=None):
        super(ResizeDialog, self).__init__(parent)
        self.setWindowTitle("Resize Image")

        self.layout = QVBoxLayout()

        self.label1 = QLabel("Width:")
        self.layout.addWidget(self.label1)
        self.width_input = QLineEdit(self)
        self.layout.addWidget(self.width_input)

        self.label2 = QLabel("Height:")
        self.layout.addWidget(self.label2)
        self.height_input = QLineEdit(self)
        self.layout.addWidget(self.height_input)

        self.button = QPushButton("Apply", self)
        self.button.clicked.connect(self.apply_resize)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

        self.new_width = None
        self.new_height = None

    def apply_resize(self):
        try:
            self.new_width = int(self.width_input.text())
            self.new_height = int(self.height_input.text())
            self.accept()
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid integers for width and height.")


class AdaptiveThresholdDialog(QDialog):
    def __init__(self, parent=None):
        super(AdaptiveThresholdDialog, self).__init__(parent)
        self.setWindowTitle("Adaptive Threshold Parameters")

        self.layout = QVBoxLayout()

        self.label1 = QLabel("Block Size (odd number):")
        self.layout.addWidget(self.label1)
        self.block_size_input = QLineEdit(self)
        self.layout.addWidget(self.block_size_input)

        self.label2 = QLabel("C (constant to subtract):")
        self.layout.addWidget(self.label2)
        self.c_input = QLineEdit(self)
        self.layout.addWidget(self.c_input)

        self.button = QPushButton("Apply", self)
        self.button.clicked.connect(self.apply_parameters)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

        self.block_size = None
        self.c = None

    def apply_parameters(self):
        try:
            self.block_size = int(self.block_size_input.text())
            self.c = int(self.c_input.text())
            if self.block_size % 2 == 1:  # Block size must be odd
                self.accept()
            else:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter a valid odd integer for block size and an integer for C.")


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
        self.seed_point = None  # To store the seed point

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
        self.show_matrices_button = self.findChild(QPushButton, 'pushButton_12')
        self.region_growing_button = self.findChild(QPushButton, 'pushButton_14')
        self.sobel_detection_button = self.findChild(QPushButton, 'pushButton_15')
        self.resize_button = self.findChild(QPushButton, 'pushButton_16')
        self.embossing_button = self.findChild(QPushButton, 'pushButton_17')
        self.sepia_effect_button = self.findChild(QPushButton, 'pushButton_18')
        self.adaptive_threshold_button = self.findChild(QPushButton, 'pushButton_19')
        self.rgb_graph = self.findChild(QPushButton, 'pushButton_20')

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

        # Set tooltips
        self.cvt2gray_button.setToolTip('Gri Tonlamaya çevir')
        self.edge_detection_button.setToolTip('Canny algoritmasını kullanarak görüntü tespiti yapma')
        self.dilation_button.setToolTip('Görüntüye genişletme (dilation) uygulama')
        self.erosion_button.setToolTip('Görüntüye erosion uygulama')
        self.opening_button.setToolTip('Görüntüye açma (opening) işlemi uygulama')
        self.closing_button.setToolTip('Görüntüye kapanma (closing) işlemi uygulama')
        self.histogram_equalization_button.setToolTip('Histogram eşitleme uygula')
        self.invert_color_button.setToolTip('Görüntünün renklerini ters çevir')
        self.simple_blur_button.setToolTip('Görüntüye basit bir bulanıklık uygula')
        self.gaussian_blur_button.setToolTip('Görüntüye Gauss bulanıklığı uygula')
        self.median_blur_button.setToolTip('Görüntüye medyan bulanıklığı uygula')
        self.bilatral_blur_button.setToolTip('Görüntüye bilateral bulanıklığı uygula')
        self.sharpend_button.setToolTip('Görüntüyü keskinleştirme')
        self.box_filter_button.setToolTip('Görüntüye bir kutu filtre uygula')
        self.show_matrices_button.setToolTip('Resim matrislerini gösterir')
        self.region_growing_button.setToolTip("Region growing segmentasyonu gerçekleştirin.")
        self.sobel_detection_button.setToolTip("Sobel kenar tespiti uygulama.")
        self.resize_button.setToolTip("Resmi yeniden boyutlandırma")
        self.embossing_button.setToolTip("Görüntüye kabartma efekti uygulama")
        self.sepia_effect_button.setToolTip("Görüntüye sepya efekti uygulama")
        self.adaptive_threshold_button.setToolTip("Görüntüye adaptif eşikleme uygulama")


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
        self.bilatral_blur_button.clicked.connect(self.bilateral_blur)

        self.simple_blur_button.clicked.connect(self.simple_blur)
        self.gaussian_blur_button.clicked.connect(self.gaussian_blur)
        self.median_blur_button.clicked.connect(self.median_blur)
        self.sharpend_button.clicked.connect(self.sharpen)
        self.box_filter_button.clicked.connect(self.box_filter)

        self.show_matrices_button.clicked.connect(self.show_matrices)
        self.sobel_detection_button.clicked.connect(self.sobel_detection)
        self.resize_button.clicked.connect(self.resize_image)
        self.embossing_button.clicked.connect(self.emboss_image)
        self.sepia_effect_button.clicked.connect(self.sepia_effect)
        self.adaptive_threshold_button.clicked.connect(self.adaptive_threshold)
        self.region_growing_button.clicked.connect(self.start_region_growing)

        self.rgb_graph.clicked.connect(self.plot_rgb_histogram_and_display_filtered_images)

        self.Apply_QToolery_Style()

        self.theme_button.clicked.connect(self.Apply_QToolery_Style)
        self.theme_button2.clicked.connect(self.Apply_QSynet_Style)
        self.theme_button3.clicked.connect(self.Apply_QDarkBlue_Style)
        self.theme_button4.clicked.connect(self.Apply_QDark_Style)

        # just for testing
        # self.erosion_button.clicked.connect(self.hello_world)

        # Initialize variables
        self.region_growing_active = False
        self.label.mousePressEvent = self.get_seed_point  # Override the mouse press event

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
        self.resize_button.setEnabled(True)
        self.adaptive_threshold_button.setEnabled(True)
        self.show_matrices_button.setEnabled(True)
        self.sobel_detection_button.setEnabled(True)
        self.embossing_button.setEnabled(True)
        self.sepia_effect_button.setEnabled(True)
        self.region_growing_button.setEnabled(True)
        self.rgb_graph.setEnabled(True)

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
        self.resize_button.setEnabled(False)
        self.adaptive_threshold_button.setEnabled(False)
        self.show_matrices_button.setEnabled(False)
        self.sobel_detection_button.setEnabled(False)
        self.embossing_button.setEnabled(False)
        self.sepia_effect_button.setEnabled(False)
        self.region_growing_button.setEnabled(False)
        self.rgb_graph.setEnabled(False)

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
            dialog = ThresholdDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                lower_threshold = dialog.lower_threshold
                upper_threshold = dialog.upper_threshold

                if lower_threshold is not None and upper_threshold is not None:
                    image = self.image_stack[-1]
                    # Convert the image to grayscale if it's not already in grayscale
                    if len(image.shape) > 2:
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_image = image

                    # Perform Canny edge detection with input thresholds
                    edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)

                    # Display the edges image
                    height, width = edges.shape
                    bytes_per_line = width
                    q_img = QImage(edges.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                    scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                                 Qt.SmoothTransformation)
                    self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

                    # Update the image stack
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
            self.display_histogram()

    def display_histogram(self):
        if self.image_stack:
            image = self.image_stack[-1]
            if image is not None:
                if len(image.shape) == 3:  # Color image
                    # Prepare a figure to plot histograms for B, G, R channels
                    plt.figure()
                    color = ('b', 'g', 'r')
                    # Manually compute histograms
                    for i, col in enumerate(color):
                        histogram = np.zeros(256)
                        for row in image:
                            for pixel in row:
                                histogram[pixel[i]] += 1
                        plt.plot(histogram, color=col)
                    plt.title('Histogram for the picture')
                    plt.xlabel('Pixel Intensity')
                    plt.ylabel('Frequency')
                    plt.show()
                elif len(image.shape) == 2:  # Grayscale image
                    histogram = np.zeros(256)
                    for row in image:
                        for pixel in row:
                            histogram[pixel] += 1
                    plt.plot(histogram, color='gray')
                    plt.title('Histogram for The image')
                    plt.xlabel('Pixel Intensity')
                    plt.ylabel('Frequency')
                    plt.show()
                else:
                    QMessageBox.warning(self, "Unsupported image format",
                                        "Loaded image format not supported for histogram display.")
            else:
                QMessageBox.warning(self, "No image", "No image loaded to display histogram.")
        else:
            QMessageBox.warning(self, "No image", "No image loaded to display histogram.")

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

    def bilateral_blur(self):
        if len(self.image_stack):
            image = self.image_stack[-1]

            # Apply Bilateral Blur
            blurred_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

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

    def show_matrices(self):
        if len(self.image_stack):
            image = self.image_stack[-1]

            # Create a new window to display the matrices
            self.matrices_window = QWidget()
            self.matrices_window.setWindowTitle("Image Matrices")

            # Convert image to string format
            if len(image.shape) == 3:  # Color image
                matrix_string = np.array2string(image, separator=', ')
            else:  # Grayscale image
                matrix_string = np.array2string(image, separator=', ')

            # Create a QLabel to display the matrices
            label = QLabel(matrix_string)
            label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            label.setWordWrap(True)

            # Create a QScrollArea
            scroll_area = QScrollArea()
            scroll_area.setWidget(label)
            scroll_area.setWidgetResizable(True)

            # Set the size of the QScrollArea
            scroll_area.setFixedSize(800, 700)

            # Create a layout and add the QScrollArea to the layout
            layout = QVBoxLayout()
            layout.addWidget(scroll_area)
            self.matrices_window.setLayout(layout)

            # Show the new window
            self.matrices_window.show()

    def sobel_detection(self):
        if self.image_stack:
            dialog = SobelDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                kernel_size = dialog.kernel_size

                if kernel_size is not None:
                    image = self.image_stack[-1]
                    # Convert the image to grayscale if it's not already in grayscale
                    if len(image.shape) > 2:
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_image = image

                    # Perform Sobel edge detection with input kernel size
                    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
                    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
                    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

                    # Convert the result back to uint8
                    sobel_combined = cv2.convertScaleAbs(sobel_combined)

                    # Display the Sobel edge detection image
                    height, width = sobel_combined.shape
                    bytes_per_line = width
                    q_img = QImage(sobel_combined.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                    scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio,
                                                 Qt.SmoothTransformation)
                    self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

                    # Update the image stack
                    self.image_stack.append(sobel_combined)
                    print(len(self.image_stack))

    def resize_image(self):
        if len(self.image_stack):
            image = self.image_stack[-1]
            dialog = ResizeDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                new_width = dialog.new_width
                new_height = dialog.new_height

                if new_width is not None and new_height is not None:
                    # Resize the image
                    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

                    # Display the resized image
                    height, width = resized_image.shape[:2]
                    if len(resized_image.shape) == 2:
                        bytes_per_line = width
                        q_img = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                    else:
                        bytes_per_line = 3 * width
                        q_img = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

                    scaled_pixmap = QPixmap.fromImage(q_img).scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.label.setPixmap(scaled_pixmap)

                    # Update the image stack
                    self.image_stack.append(resized_image)
                    self.get_size()  # Update the width and height labels
                    print(len(self.image_stack))
                else:
                    QMessageBox.warning(self, "Input Error", "Width and height cannot be None.")


    def emboss_image(self):
        if len(self.image_stack):
            image = self.image_stack[-1]

            # Convert the image to grayscale if it's not already in grayscale
            if len(image.shape) > 2:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Define the emboss kernel
            kernel = np.array([[-2, -1, 0],
                               [-1, 1, 1],
                               [0, 1, 2]])

            # Apply the emboss filter
            embossed_image = cv2.filter2D(gray_image, -1, kernel)

            # Add 128 to shift the intensity values to the visible range
            embossed_image = cv2.add(embossed_image, 128)

            # Display the embossed image
            height, width = embossed_image.shape
            bytes_per_line = width
            q_img = QImage(embossed_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(self.label.size(), Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

            # Update the image stack
            self.image_stack.append(embossed_image)
            print(len(self.image_stack))

    def sepia_effect(self):
        if len(self.image_stack):
            image = self.image_stack[-1]

            # Convert the image to BGR if it's not already
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Apply sepia filter
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            sepia_image = cv2.transform(image, kernel)
            sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

            # Display the sepia image
            height, width, channel = sepia_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(sepia_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(self.label.size(), Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

            # Update the image stack
            self.image_stack.append(sepia_image)
            print(len(self.image_stack))

    def adaptive_threshold(self):
        if self.image_stack:
            image = self.image_stack[-1]
            dialog = AdaptiveThresholdDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                block_size = dialog.block_size
                c = dialog.c

                if block_size is not None and c is not None and block_size % 2 == 1 and block_size > 1:
                    if len(image.shape) > 2:
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_image = image

                    adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                            cv2.THRESH_BINARY, block_size, c)

                    height, width = adaptive_thresh.shape
                    bytes_per_line = width
                    q_img = QImage(adaptive_thresh.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                    scaled_pixmap = q_img.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.label.setPixmap(QPixmap.fromImage(scaled_pixmap))

                    self.image_stack.append(adaptive_thresh)
                    print("Stack size after adding:", len(self.image_stack))
                else:
                    print("Invalid block_size or c value:", block_size, c)
        else:
            print("No images in stack to process.")

    def start_region_growing(self):
        self.region_growing_active = True
        QMessageBox.information(self, 'Region Growing', 'Click on the image to select a seed point.')

    def get_seed_point(self, event):
        if self.region_growing_active:
            x = event.pos().x()
            y = event.pos().y()
            self.seed_point = (x, y)
            print(f"x: {x}\ny: {y}")
            self.region_growing_active = False
            self.apply_region_growing()

    def apply_region_growing(self):
        if self.seed_point and self.image_stack:
            image = self.image_stack[-1]
            if len(image.shape) == 3:  # Check if it's a BGR image
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            seed_x, seed_y = self.seed_point
            seed_x = int(seed_x * (gray_image.shape[1] / self.label.width()))
            seed_y = int(seed_y * (gray_image.shape[0] / self.label.height()))
            region = self.region_grow(gray_image, (seed_y, seed_x))

            color_image = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
            height, width, channel = color_image.shape
            bytes_per_line = width * channel
            q_img = QImage(color_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)
            self.image_stack.append(region)

    def region_grow(self, img, seed, threshold=30):
        h, w = img.shape
        mask = np.zeros((h, w), np.uint8)
        mask[seed[0], seed[1]] = 255
        seed_value = img[seed[0], seed[1]]

        queue = [seed]
        while queue:
            x, y = queue.pop(0)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] == 0:
                    if abs(int(img[nx, ny]) - int(seed_value)) < threshold:
                        mask[nx, ny] = 255
                        queue.append((nx, ny))

        return mask

    def plot_rgb_histogram(self):
        if len(self.image_stack):
            image = self.image_stack[-1]

            # Ensure the image is in RGB format
            if len(image.shape) == 2 or image.shape[2] == 1:
                # Convert grayscale to RGB if needed
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Calculate histograms for each color channel
            color = ('r', 'g', 'b')
            for i, col in enumerate(color):
                hist = cv2.calcHist([rgb_image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])

            plt.title('RGB Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.show()

            # Optionally, you can add the RGB image to the stack or update the GUI to display the RGB image
            self.image_stack.append(rgb_image)
            print("Updated image stack with RGB image.")

    def plot_rgb_histogram_and_display_filtered_images(self):
        if len(self.image_stack):
            image = self.image_stack[-1]

            # Ensure the image is in RGB format
            if len(image.shape) == 2 or image.shape[2] == 1:
                # Convert grayscale to RGB if needed
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Calculate histograms for each color channel
            color = ('r', 'g', 'b')
            for i, col in enumerate(color):
                hist = cv2.calcHist([rgb_image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])

            plt.title('RGB Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.show()

            # Create filtered images
            red_filtered = rgb_image.copy()
            red_filtered[:, :, 1] = 0  # Zero out the green channel
            red_filtered[:, :, 2] = 0  # Zero out the blue channel

            green_filtered = rgb_image.copy()
            green_filtered[:, :, 0] = 0  # Zero out the red channel
            green_filtered[:, :, 2] = 0  # Zero out the blue channel

            blue_filtered = rgb_image.copy()
            blue_filtered[:, :, 0] = 0  # Zero out the red channel
            blue_filtered[:, :, 1] = 0  # Zero out the green channel

            # Display the filtered images in new windows

            cv2.imshow('Green Filtered Image', green_filtered)
            cv2.imshow('Blue Filtered Image', blue_filtered)
            cv2.imshow('Red Filtered Image', red_filtered)
            cv2.waitKey(0)  # Wait for a key press to close the images
            cv2.destroyAllWindows()

            # Optionally, you can add the filtered images to the stack
            self.image_stack.extend([red_filtered, green_filtered, blue_filtered])
            print("Updated image stack with filtered images.")


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
