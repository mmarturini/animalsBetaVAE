import os
from google.colab import drive

folder_mount = '/content/drive' 
drive.mount(folder_mount)
os.chdir("/content/drive/MyDrive/DSSC/Deep_Learning/Lab_Notebooks")
print(os.getcwd())