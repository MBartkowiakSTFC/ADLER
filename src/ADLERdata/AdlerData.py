
from PyQt6.QtCore import QDir
from PyQt6.QtGui import QIcon

class AdlerData():
            
    def __init__(self, path):

        self.res_dir = QDir(path)
        self._files = {}
        self.res_dir.setNameFilters(["*.dat","*.txt"])
        files = self.res_dir.entryList()
        for f in files:
            label = ".".join(str(f).split('.')[:-1])
            self._files[label] = self.res_dir.filePath(f)
        
DATA = AdlerData('.')     
