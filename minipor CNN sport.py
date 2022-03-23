from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QFileDialog
from pip import main
from sport_ui import Ui_MainWindow
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
from PyQt5.QtGui import QPixmap
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import random
import os
import sys
import shutil
from pathlib import Path
import imghdr


class finalexam(Ui_MainWindow):
    def __init__(self) -> None:
        super().setupUi(MainWindow)
        os.chdir(sys.path[0])
        self.model=models.load_model("minipro1.h5")
        self.class_list=['badminton', 'baseball', 'basketball', 'fencing', 'football']
        self.gcn()
        self.i = 1
        self.x = 0
        self.plainTextEdit.appendPlainText("Please select a folder")
        self.Checkbt.setEnabled(False)
        self.renamebt.setEnabled(False)
        self.refreshbt.setEnabled(False)
        self.nextbt.setEnabled(False)

    def gcn(self):
        self.browsbt.clicked.connect(self.browsclick)
        self.nextbt.clicked.connect(self.nextclick)
        self.renamebt.clicked.connect(self.renameclick)   
        self.refreshbt.clicked.connect(self.refreshclick) 
        self.Checkbt.clicked.connect(self.checfunc)
    
    def checfunc(self):
        self.data_dir = self.file_path
        self.image_extensions = [".png", ".jpg"]  # add there all your images file extensions

        self.img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
        for self.filepath in Path(self.data_dir).rglob("*"):
            if self.filepath.suffix.lower() in self.image_extensions:
                self.img_type = imghdr.what(self.filepath)
                if self.img_type is None:
                    print(f"{self.filepath} is not an image")
                    self.plainTextEdit.appendPlainText(f"{self.filepath} is not an image")
                elif self.img_type not in self.img_type_accepted_by_tf:
                    print(f"{self.filepath} is a {self.img_type}, not accepted by TensorFlow")
                    self.plainTextEdit.appendPlainText(self.plainTextEdit.appendPlainText(f"{self.filepath} is a {self.img_type}, not accepted by TensorFlow"))
        self.plainTextEdit.appendPlainText("Finished!")

    def refreshclick(self):
        self.listpic=os.listdir(self.file_path)
        self.nextbt.setEnabled(True)
        self.plainTextEdit.appendPlainText("Refreshed!")

    def nextclick(self):
        # self.listpic=os.listdir(self.file_path)
        self.lpig=len(self.listpic)
        print(self.listpic)
        print(self.lpig)
        # self.randpic=random.choice(self.listpic)
        if self.lpig > 0:
            self.randpic=self.listpic[0]
            self.pixmap=QPixmap(self.file_path+"\\"+self.randpic)
            self.labelimg.setScaledContents(True)
            self.labelimg.setPixmap(self.pixmap)
            self.picpred()
            self.class_display.setText(self.predlabel)
            self.plainTextEdit.appendPlainText(self.randpic+" : "+ self.predlabel)
            self.listpic.pop(0)
        elif self.lpig == 0:
            self.plainTextEdit.appendPlainText("Done!,If you want to predict again, Please \"press Refresh button\"")

    def renameclick(self):
        self.listpic=os.listdir(self.file_path)
        for self.randpic in self.listpic:
            self.picpred()
            os.rename(self.file_path+"\\"+self.randpic,self.file_path+"\\"+self.predlabel+str(self.i)+".jpeg")
            self.i+=1
            self.plainTextEdit.appendPlainText(self.predlabel)
        self.plainTextEdit.appendPlainText("Finished!")
        self.plainTextEdit.appendPlainText("Please press \"Refresh button\"")
        self.nextbt.setEnabled(False)
    

    def browsclick(self):
        dialog=QFileDialog()
        self.file_path=dialog.getExistingDirectory(None,"Select Folder")
        # print(self.file_path)
        self.lineEdit.setText(self.file_path)
        self.listpic=os.listdir(self.file_path)
        self.plainTextEdit.appendPlainText("Please press \"Check button\" to check the image file")
        self.Checkbt.setEnabled(True)
        self.renamebt.setEnabled(True)
        self.refreshbt.setEnabled(True)
        self.nextbt.setEnabled(True)
    
    def picpred(self):
        self.multiclass_pred_plot(self.model,(self.file_path+"\\"+self.randpic),
        self.class_list,
        img_shape=180) 

    def multiclass_pred_plot(self,model, filename, class_names, img_shape=180):
        self.img = tf.io.read_file(filename)
        self.img = tf.image.decode_image(self.img, channels=3)
        self.img = tf.image.resize(self.img, size = [img_shape, img_shape])
        self.pred = self.model.predict(tf.expand_dims(self.img, axis=0))
        self.pred_class=self.pred.argmax(axis=1)
        self.predlabel=self.class_list[self.pred_class[0]]
        print(self.predlabel)

if __name__ == "__main__":
    ui = finalexam()
    MainWindow.show()
    sys.exit(app.exec_())
