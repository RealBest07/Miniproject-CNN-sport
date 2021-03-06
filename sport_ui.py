# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sport_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(920, 674)
        MainWindow.setStyleSheet("background-color: rgb(210, 255, 225);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(14, 15, 891, 91))
        font = QtGui.QFont()
        font.setFamily("Sitka Text Semibold")
        font.setPointSize(37)
        font.setBold(True)
        font.setWeight(75)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.label.setFont(font)
        self.label.setStyleSheet("background-color: rgb(170, 255, 127);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 180, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("background-color: rgb(196, 246, 255);")
        self.label_2.setObjectName("label_2")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(530, 250, 351, 291))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.plainTextEdit.setPlainText("")
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.class_display = QtWidgets.QLabel(self.centralwidget)
        self.class_display.setGeometry(QtCore.QRect(220, 180, 301, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.class_display.setFont(font)
        self.class_display.setStyleSheet("background-color: rgb(111, 255, 183);")
        self.class_display.setObjectName("class_display")
        self.labelimg = QtWidgets.QLabel(self.centralwidget)
        self.labelimg.setGeometry(QtCore.QRect(30, 250, 491, 351))
        self.labelimg.setStyleSheet("background-color: rgb(255, 250, 194);")
        self.labelimg.setAlignment(QtCore.Qt.AlignCenter)
        self.labelimg.setObjectName("labelimg")
        self.renamebt = QtWidgets.QPushButton(self.centralwidget)
        self.renamebt.setGeometry(QtCore.QRect(530, 550, 351, 51))
        font = QtGui.QFont()
        font.setFamily("Nikkyou Sans")
        font.setPointSize(14)
        self.renamebt.setFont(font)
        self.renamebt.setStyleSheet("background-color: rgb(170, 255, 255);")
        self.renamebt.setObjectName("renamebt")
        self.browsbt = QtWidgets.QPushButton(self.centralwidget)
        self.browsbt.setGeometry(QtCore.QRect(530, 140, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Nikkyou Sans")
        font.setPointSize(10)
        self.browsbt.setFont(font)
        self.browsbt.setStyleSheet("background-color: rgb(255, 255, 127);")
        self.browsbt.setObjectName("browsbt")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(20, 110, 201, 21))
        font = QtGui.QFont()
        font.setFamily("Nikkyou Sans")
        font.setPointSize(10)
        self.label_16.setFont(font)
        self.label_16.setStyleSheet("")
        self.label_16.setObjectName("label_16")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(20, 140, 501, 31))
        self.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.nextbt = QtWidgets.QPushButton(self.centralwidget)
        self.nextbt.setGeometry(QtCore.QRect(530, 180, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Nikkyou Sans")
        font.setPointSize(16)
        self.nextbt.setFont(font)
        self.nextbt.setStyleSheet("background-color: rgb(85, 255, 127);")
        self.nextbt.setObjectName("nextbt")
        self.refreshbt = QtWidgets.QPushButton(self.centralwidget)
        self.refreshbt.setGeometry(QtCore.QRect(710, 180, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Nikkyou Sans")
        font.setPointSize(15)
        self.refreshbt.setFont(font)
        self.refreshbt.setStyleSheet("background-color: rgb(255, 170, 255);")
        self.refreshbt.setObjectName("refreshbt")
        self.Checkbt = QtWidgets.QPushButton(self.centralwidget)
        self.Checkbt.setGeometry(QtCore.QRect(710, 140, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Nikkyou Sans")
        font.setPointSize(10)
        self.Checkbt.setFont(font)
        self.Checkbt.setStyleSheet("background-color: rgb(255, 99, 99);")
        self.Checkbt.setObjectName("Checkbt")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 920, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Sports Classify"))
        self.label_2.setText(_translate("MainWindow", "Prediction Result"))
        self.class_display.setText(_translate("MainWindow", "objest class"))
        self.labelimg.setText(_translate("MainWindow", "TextLabel"))
        self.renamebt.setText(_translate("MainWindow", "Rename All"))
        self.browsbt.setText(_translate("MainWindow", "Browse"))
        self.label_16.setText(_translate("MainWindow", "Image Folder"))
        self.nextbt.setText(_translate("MainWindow", "Next"))
        self.refreshbt.setText(_translate("MainWindow", "Refresh"))
        self.Checkbt.setText(_translate("MainWindow", "Check"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

