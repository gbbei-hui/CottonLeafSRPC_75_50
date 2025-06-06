
import os
import glob



# 读取所有文件，并且重新命名

def ReadDir(DirPath,FileType):
    List_AllFile = glob.glob(DirPath + "/*"+ FileType)
    print("File_n:", len(List_AllFile))
    return List_AllFile