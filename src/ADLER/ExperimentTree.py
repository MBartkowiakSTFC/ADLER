#    This file is part of ADLER.
#
#    ADLER is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright (C) Maciej Bartkowiak, 2019-2023

import numpy as np
import yaml
import copy
import os
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

# from PyQt6.QtCore import QMutex,  QThread

from PyQt6.QtCore import (
    QMimeData,
    Qt,
    QRectF,
    QRect,
    QPointF,
    QPoint,
    QDataStream,
    QByteArray,
    QIODevice,
)
from PyQt6.QtGui import (
    QDrag,
    QTransform,
    QPen,
    QPainter,
    QBrush,
    QFontMetricsF,
    QColor,
    QPixmap,
)
from PyQt6.QtGui import QColorConstants as QCC
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsView,
    QDialog,
    QWidget,
    QSizePolicy,
    QGraphicsItem,
    QMenu,
    QTableView,
    QScrollArea,
    QTabWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QComboBox,
    QFrame,
    QLabel,
)
from PyQt6 import sip

from ADLER.ExtendGUI import LogBox, ManyButtons, PatternDialog
from ADLER.LogviewerTab import LogviewerTab
from ADLER.NewSingleTab import SingleTab
from ADLER.FileFinder import header_read
from ADLER.ADLERcalc.spectrumUtils import precise_merge
from ADLER.ADLERcalc.ioUtils import read_1D_curve, simplify_number_range, resource_path

# from geom_tools import normalise, length, arb_rotation

last_object = 1


class MeasPointMergeDialog(QDialog):
    values_ready = pyqtSignal(object)

    def __init__(self, parent, dnode):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.rightside = QWidget(self)
        self.rightlayout = QVBoxLayout(self.rightside)
        # mb1 = ManyButtons(self, ['FWHM', 'Area', 'Centre'], label = "Channels",  ticked = False)
        # mb2 = ManyButtons(self, ['FWHM', 'Area', 'Centre'], label = "Energy",  ticked = False)
        # mb1.new_values.connect(self.take3)
        # mb2.new_values.connect(self.take36)
        button1 = QPushButton("Overplot", self)
        button1.clicked.connect(self.overplot_curves)
        button2 = QPushButton("Merge", self)
        button2.clicked.connect(self.merge_curves)
        cb1 = QComboBox(self)
        self.combo = cb1
        # self.rightlayout.addWidget(mb1)
        # self.rightlayout.addWidget(mb2)
        self.rightlayout.addWidget(cb1)
        self.rightlayout.addWidget(button1)
        self.rightlayout.addWidget(button2)
        can, fig, lay = self.MakeCanvas(self)
        self.layout.addWidget(can)
        self.layout.addWidget(self.rightside)
        self.canvas = can
        self.figure = fig
        self.datanode = dnode
        self.populateCbox()
        self.setWindowTitle("Compare files in Measurement Point " + self.datanode.name)

    def populateCbox(self):
        self.combo.addItem("Channels", [])
        self.combo.addItem("Energy [eV]", [])

    def MakeCanvas(self, parent):
        mdpi, winch, hinch = 75, 9.0 * mpl_figure_scale, 7.0 * mpl_figure_scale
        canvas = QWidget(parent)
        layout = QVBoxLayout(canvas)
        figure = mpl.figure(figsize=[winch, hinch], dpi=mdpi)  # , frameon = False)
        figAgg = FigureCanvasQTAgg(figure)
        figAgg.setParent(canvas)
        figAgg.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        figAgg.updateGeometry()
        toolbar = NavigationToolbar2QTAgg(figAgg, canvas)
        toolbar.update()
        layout.addWidget(figAgg)
        layout.addWidget(toolbar)
        return canvas, figure, layout

    def prepare_curves(self):
        xlabel = self.combo.currentText()
        if xlabel == "Channels":
            curve_names = self.datanode.returnCurves()
            suffix = "_1D.txt"
        elif xlabel == "Energy [eV]":
            curve_names = self.datanode.returnEnergyCurves()
            suffix = "_1D_deltaE.txt"
        else:
            print("Not a valid unit for plotting! What have you done?")
            return None
        if len(curve_names) < 1:
            print(
                "Did not find any curves in this data node. Try creating them first with ADLER."
            )
            return None
        clabels = []
        curves = []
        unitlist = []
        energylist = []
        fnames = []
        fpaths = []
        for n, entry in enumerate(curve_names):
            a, b, c = read_1D_curve(entry[1])
            curves.append(a)
            energylist.append(b)
            unitlist.append(c)
            label = str(b[0]) + "$\\pm$" + str(b[-1]) + " eV, " + entry[0]
            clabels.append(label)
            fnames.append(entry[1])
            fpaths.append("/".join(entry[1].split("/")[:-1]))
        unitmask = []
        for u in unitlist:
            if "Transfer" in u:
                unitmask.append(2)
            elif "Energy" in u:
                unitmask.append(1)
            else:
                unitmask.append(0)
        unitmask = np.array(unitmask)
        unitmax = unitmask.max()
        boolmask = unitmask == unitmax
        finalcurves, finallabels = [], []
        finalnames = []
        for n, u in enumerate(boolmask):
            if u:
                finalcurves.append(curves[n])
                finallabels.append(clabels[n])
                finalnames.append(fnames[n])
        self.fcurves = finalcurves
        self.flabels = finallabels
        self.fnames = finalnames
        tempp = np.unique(fpaths)
        if len(tempp) > 1:
            print(
                "Multiple input paths in MeasPoint; the first one will be used for output.\n",
                tempp,
            )
        self.output_path = tempp[0]
        self.output_name = (
            "_".join(["BestMerge"] + simplify_number_range(finalnames)) + suffix
        )

    def overplot_curves(self):
        self.prepare_curves()
        xlabel = self.combo.currentText()
        plot1D(
            self.fcurves,
            outFile="",
            fig=self.figure,
            text="",
            label_override=[xlabel, ""],
            curve_labels=self.flabels,
            title="Overplot curves in " + self.datanode.name,
            autolimits=False,
        )

    def merge_curves(self):
        self.prepare_curves()
        xlabel = self.combo.currentText()
        output = self.output_path + "/" + self.output_name
        merged_curve = precise_merge(self.fnames, output)
        plot1D(
            [merged_curve],
            outFile="",
            fig=self.figure,
            text="",
            label_override=[xlabel, ""],
            curve_labels=self.output_name,
            title="Merged curves from " + self.datanode.name,
            autolimits=False,
        )


class MeasPointDialog(PatternDialog):
    def __init__(self, parent, datanode):
        super().__init__(parent, dialogname="Merging tool for " + datanode.name)


class AdlerDialog(QDialog):
    values_ready = pyqtSignal(object)

    def __init__(self, parent, dnode, appinstance=None):
        super().__init__(parent)
        self.enpars, self.fitpars, self.adlerpars = None, None, None
        self.dnode = dnode  # this has to be a SingleFile node
        fpath, fname = dnode.fpath, dnode.filename
        spath = "/".join(fpath.split("/")[:-1])
        self.layout = QVBoxLayout()
        self.tabbar = QTabWidget(self)
        self.tabbar.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding
        )
        self.button = QPushButton("Accept Changes", self)
        self.button.clicked.connect(self.send_values_back)
        logtab = self.LogTab()
        xastab = LogviewerTab(
            self.tabbar, self.MakeCanvas(self.tabbar), self.log, startpath=spath
        )
        datatab = SingleTab(
            self.tabbar,
            self.MakeCanvas(self.tabbar),
            self.log,
            mthreads=1,
            startpath=spath,
            logplotter=xastab,
            app=appinstance,
            dialog=True,
            accept_button=self.button,
        )
        self.tabbar.addTab(datatab.base, "Data reduction")
        self.tabbar.addTab(xastab.base, "XAS file viewer")
        self.tabbar.addTab(logtab, "ADLER Log")
        self.layout.addWidget(self.tabbar)
        # self.layout.addWidget(self.button)
        self.setLayout(self.layout)
        self.setWindowTitle("Pop-up ADLER for " + fname)
        datatab.load_params_from_dict(self.dnode.ADLER_params)
        datatab.core.get_external_params(self.dnode.ADLER_params)
        datatab.core.params_fitting.connect(self.take_fitting_results)
        datatab.core.params_energy.connect(self.take_energy_results)
        self.coreref = datatab.core
        datatab.filelist = [fpath]
        datatab.reload_file()

    def MakeCanvas(self, parent):
        mdpi, winch, hinch = 75, 9.0 * mpl_figure_scale, 7.0 * mpl_figure_scale
        canvas = QWidget(parent)
        layout = QVBoxLayout(canvas)
        figure = mpl.figure(figsize=[winch, hinch], dpi=mdpi)  # , frameon = False)
        figAgg = FigureCanvasQTAgg(figure)
        figAgg.setParent(canvas)
        figAgg.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        figAgg.updateGeometry()
        toolbar = NavigationToolbar2QTAgg(figAgg, canvas)
        toolbar.update()
        layout.addWidget(figAgg)
        layout.addWidget(toolbar)
        return canvas, figure, layout

    def LogTab(self):
        base = QWidget()
        base.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        base_layout = QVBoxLayout()
        # layout = QFormLayout()
        box = LogBox(base)
        self.log = box
        box.setReadOnly(True)
        base_layout.addWidget(box)
        base.setLayout(base_layout)
        return base

    @pyqtSlot(object)
    def take_fitting_results(self, results):
        self.fitpars = results
        self.adlerpars = copy.deepcopy(self.coreref.pardict)
        # print("New fitting:",  fitpars)

    @pyqtSlot(object)
    def take_energy_results(self, results):
        self.enpars = results
        self.adlerpars = copy.deepcopy(self.coreref.pardict)
        # print("New energy fitting:",  fitpars)

    def send_values_back(self):
        tempdict = {}
        for dd in [self.adlerpars, self.fitpars, self.enpars]:
            if dd is not None:
                for kk in dd.keys():
                    tempdict[kk] = dd[kk]
        self.values_ready.emit([None, tempdict])
        self.done(0)


class DataDialog(QDialog):
    values_ready = pyqtSignal(object)

    def __init__(self, parent, dnode):
        super().__init__(parent)
        self._dataNode = dnode
        self.base = QWidget(self)
        # self.scroller = QScrollArea()
        # self.scroller.setWidget(self.base)
        self.layout = QVBoxLayout()
        self.sublayout = QHBoxLayout(self.base)
        self.leftside = QWidget(self.base)
        self.rightside = QWidget(self.base)
        scroller1 = QScrollArea(widgetResizable=True)
        scroller1.setWidget(self.leftside)
        scroller2 = QScrollArea(widgetResizable=True)
        scroller2.setWidget(self.rightside)
        self.leftform = QFormLayout(self.leftside)
        self.rightform = QFormLayout(self.rightside)
        # self.layout.addWidget(self.leftside)
        # self.layout.addWidget(self.rightside)
        self.sublayout.addWidget(scroller1)
        self.sublayout.addWidget(scroller2)
        self.ldict = dnode.package_data()
        self.rdict = dnode.extra_data()
        self.lfields = {}
        self.rfields = {}
        print("Elements in node data: ", len(self.ldict))
        print("Elements in extra data: ", len(self.rdict))
        for kk in self.ldict.keys():
            key = str(kk)
            value = str(self.ldict[kk])
            tfield = QLineEdit(value, parent=self)
            self.lfields[key] = tfield
            self.leftform.addRow(key, tfield)
        for kk in self.rdict.keys():
            key = str(kk)
            value = str(self.rdict[kk])
            tfield = QLineEdit(value, parent=self)
            self.rfields[key] = tfield
            self.rightform.addRow(key, tfield)
        self.layout.addWidget(self.base)
        self.button = QPushButton("Accept Changes", self)
        self.button.clicked.connect(self.send_values_back)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)
        self.setWindowTitle("Node Property Editor")

    def send_values_back(self):
        leftdict = {}
        rightdict = {}
        for kk in self.lfields.keys():
            key = str(kk)
            text = self.lfields[kk].text()
            leftdict[key] = text
        for kk in self.rfields.keys():
            key = str(kk)
            text = self.rfields[kk].text()
            rightdict[key] = text
        self.values_ready.emit([leftdict, rightdict])
        self.done(0)


class DataNode(QObject):
    """
    DataNode is an abstract object in the PEAXIS ExperimentTree.
    It can have parents, children, and it is able to store information
    about itselt into a YAML file.
    """

    def __init__(self, placetag=""):
        super().__init__()
        self.data = []
        self.metadata = {}
        self.name = "NULL"
        self.type = "None"
        self.place = "/"
        self.placetag = placetag
        self.children = []
        self.child_ptrs = []
        self.child_number = 1
        self.parent = None
        self.parent_ptr = None
        self.plotNode = None

    def save(self, yamlfile):
        datadict = self.package_data()
        extra = self.extra_data()
        for kk in extra.keys():
            datadict[kk] = extra[kk]
        yaml.dump(datadict, stream=yamlfile)
        yamlfile.write("---\n")
        for ch in self.child_ptrs:
            ch.save(yamlfile)

    def fullpath(self):
        root = ""
        if self.parent_ptr is not None:
            root = self.parent_ptr.fullpath() + "/"
        else:
            root = "/"
        self.place = root + self.placetag
        return self.place

    def extra_data(self):
        return {}

    def next_number(self):
        nnum = int(self.child_number)
        self.child_number += 1
        return nnum

    def registerNode(self, temp):
        """
        This function registers the new node in the list of all the nodes.
        The list is kept by the head node.
        """
        if self.parent_ptr is not None:
            self.parent_ptr.registerNode(temp)
        else:
            print(
                "Something went wrong: trying to register a DataNode that has no parent."
            )

    def unregisterNode(self, temp):
        """
        This function registers the new node in the list of all the nodes.
        The list is kept by the head node.
        """
        if self.parent_ptr is not None:
            self.parent_ptr.unregisterNode(temp)

    def addChild(self, newPoint):
        """
        This function is meant to be used as the frontend for
        changing ownership of the nodes.
        So, to move a node to a new parent, it should be enough
        to run:
        parent.addChild(datanode)
        """
        try:
            type = newPoint.type
        except:
            return False
        else:
            if newPoint not in self.child_ptrs:
                # first we clean up the old connections
                newPoint.unregisterNode(newPoint)
                newPoint.unsetParent()
                # then we add the node to the existing one
                self.child_ptrs.append(newPoint)
                newPoint.setParent(self)
                self.children.append(newPoint.fullpath())
                self.registerNode(newPoint)
                return True
            else:
                return False
            # self.registerNode([newPoint, newPoint.type,  newPoint.name,  newPoint.place])

    def deleteChild(self, dnode):
        if dnode in self.child_ptrs:
            dnode.unregisterNode(dnode)
            dnode.unsetParent()

    def setName(self, newname):
        self.name = newname

    def setPlace(self, newplace):
        self.placetag = newplace
        print(self.fullpath())

    def unsetParent(self):
        if self.parent_ptr is not None:
            oldpath = self.fullpath()
            if self in self.parent_ptr.child_ptrs:
                ind = self.parent_ptr.child_ptrs.index(self)
                self.parent_ptr.child_ptrs.pop(ind)
            if oldpath in self.parent_ptr.children:
                ind = self.parent_ptr.children.index(oldpath)
                self.parent_ptr.children.pop(ind)
            self.parent_ptr = None
            self.parent = "None"

    def setParent(self, dnode):
        self.parent_ptr = dnode
        self.setPlace(self.placetag)
        self.parent = dnode.fullpath()
        self.fullpath()

    def package_data(self):
        tempdict = {}
        tempdict["name"] = self.name
        tempdict["type"] = self.type
        tempdict["place"] = self.fullpath()
        tempdict["placetag"] = self.placetag
        if self.parent_ptr is not None:
            tempdict["parent"] = self.parent_ptr.fullpath()
        else:
            tempdict["parent"] = None
        tempdict["children"] = []
        for ch in self.child_ptrs:
            tempdict["children"].append(ch.fullpath())
        tempdict["metadata"] = self.metadata
        tempdict["data"] = self.data
        return tempdict

    def take_params(self, tempdict):
        self.name = tempdict["name"]
        self.type = tempdict["type"]
        self.children = []
        for ch in tempdict["children"]:
            self.children.append(ch)
        self.metadata = tempdict["metadata"]
        self.data = tempdict["data"]
        self.place = tempdict["place"]
        self.parent = tempdict["parent"]

    def take_other_vars(self, tempdict):
        pass

    @pyqtSlot(object)
    def accept_values(self, newdicts):
        if newdicts[0] is not None:
            self.take_params(newdicts[0])
        if newdicts[1] is not None:
            self.take_other_vars(newdicts[1])

    def num_children(self):
        return len(self.children)

    def assignPlotNode(self, nnode):
        self.plotNode = nnode

    def findHeadNode(self):
        if self.parent is not None:
            return self.parent.findHeadNode()
        else:
            return self


class Experiment(DataNode):
    """
    An Experiment is meant to store information about
    all the actions performed as a part of a proposal
    on the PEAXIS instrument.
    """

    def __init__(self, rootdir=".", placetag=""):
        super().__init__(placetag=placetag)
        self.rootdir = rootdir
        self.type = "Experiment"
        self.name = "NULL"
        self.place = "/root"
        self.placetag = "root"
        self.all_nodes = []
        self.restore()

    def extra_data(self):
        tempdict = {}
        tempdict["rootdir"] = self.rootdir
        return tempdict

    def SaveAll(self):
        tname = os.path.join(self.rootdir, "main.yaml")
        target = open(tname, "w")
        self.save(target)
        target.close()

    def registerNode(self, temp):
        name_match = False
        place_match = False
        for n in self.all_nodes:
            if n[2] == temp.name:
                name_match = True
            if n[3] == temp.place:
                place_match = True
        if name_match and place_match:
            return None
        else:
            self.all_nodes.append([temp, temp.type, temp.name, temp.place])

    def createChild(self, pardict):
        nodetype = pardict["type"]
        if nodetype == "Experiment":
            self.take_params(pardict)
            self.registerNode(self)
        elif nodetype == "Holder":
            temp = SampleHolder(placetag=pardict["placetag"])
            temp.take_params(pardict)
            self.registerNode(temp)
        elif nodetype == "Sample":
            temp = Sample(placetag=pardict["placetag"])
            temp.take_params(pardict)
            self.registerNode(temp)
        elif nodetype == "Measurement":
            temp = MeasPoint(placetag=pardict["placetag"])
            temp.take_params(pardict)
            self.registerNode(temp)
        elif nodetype == "SingleFile":
            temp = SingleFile(placetag=pardict["placetag"])
            temp.take_params(pardict)
            self.registerNode(temp)

    def restore(self, name="main.yaml"):
        tname = os.path.join(self.rootdir, name)
        try:
            target = open(tname, "r")
        except:
            return False
        else:
            gen = yaml.load_all(target, Loader=yaml.Loader)
            for ch in gen:
                # print('#',ch)
                if ch is not None:
                    self.createChild(ch)
            target.close()
            self.reconstructTree()
            return True

    def reconstructTree(self):
        nesting = []
        expnum = []
        measnum = []
        filenum = []
        for num, n in enumerate(self.all_nodes):
            nesting.append(len(n[3].split("/")))
            if n[1] == "Experiment":
                expnum.append(num)
            elif n[1] == "Measurement":
                measnum.append(num)
            elif n[1] == "SingleFile":
                filenum.append(num)
        nesting = np.array(nesting)
        nmin = nesting.min()
        nmax = nesting.max()
        # root = self.all_nodes[np.where(nesting==nmin)][0]
        for nnest in np.arange(nmin + 1, nmax + 1):
            parents = []
            inds = np.where(nesting == (nnest - 1))[0]
            for x in inds:
                parents.append(self.all_nodes[x])
            for num, n in enumerate(self.all_nodes):
                if nesting[num] == nnest:
                    place = n[3]
                    parent_place = "/".join(place.split("/")[:-1])
                    for p in parents:
                        print(p[3])
                        if p[3] == parent_place:
                            # n[0].setParent(p[0])
                            p[0].addChild(n[0])


class SampleHolder(DataNode):
    """
    Following the structure of an experiment on PEAXIS,
    SampleHolder comes directly after Experiment in the
    experiment tree. There can be many samples on a single
    holder, but the calibration measurements are frequently
    transferable between samples on the same holder.
    """

    def __init__(self, placetag=""):
        super().__init__(placetag=placetag)
        self.type = "Holder"
        self.holder_label = "Holder001"
        self.samples = []
        self.calibration = []
        self.holder_map = ""
        self.metadata = {}

    def extra_data(self):
        tdict = {}
        return tdict

    def addFile(self, newFile):
        self.comp_files.append(newFile)

    def deleteFile(self, index):
        self.comp_files.pop(index)


class SampleComposition:
    def __init__(self, formula=""):
        self.formula = formula
        self.parse_formula()

    def parse_formula(self):
        self.elements = []
        self.composition = {}
        self.isotopes = {}
        toks = self.formula.split(":")
        for t in toks:
            nchar = len(t)
            iso = -1
            conc = 0.0
            for x in np.arange(nchar):
                try:
                    iso = int(t[:x])
                except:
                    pass
                try:
                    conc = float(t[: -x - 1])
                except:
                    pass
            chemname = t.strip("0123456789.-")
            if conc > 0:
                self.elements.append(chemname)
                self.composition[chemname] = conc
                self.isotopes[chemname] = iso

    def set_formula(self, newform):
        self.formula = newform
        self.parse_formula()


class SampleOrientation:
    def __init__(self, uvec=np.array([1.0, 0.0, 0.0]), vvec=np.array([0.0, 0.0, 1.0])):
        self.U = []
        self.Uinv = []
        self.repang = None
        self.repvec = None
        self.u, self.v, self.w = uvec, vvec, None
        self.rotmat = None
        self.step = None
        self.reciprocal_vectors()
        self.themat = []
        self.unitcell = None

    def assignUnitcell(self, ucell):
        self.unitcell = ucell

    def makeU(self, u=np.array([1.0, 0.0, 0.0]), v=np.array([0.0, 1.0, 0.0])):
        if self.unitcell is None:
            return None
        else:
            b = self.unitcell.B
        bu = normalise(np.dot(b, u))
        bv = normalise(np.dot(b, v))
        if (length(bu) < 1e-6) or (length(bv) < 1e-6):
            return False
        bw = normalise(np.cross(bu, bv))
        if length(bw) < 1e-6:
            return False
        bv = normalise(np.cross(bw, bu))
        lab = np.zeros((3, 3))
        lab[0, 1] = -1.0
        lab[1, 0] = 1.0
        lab[2, 2] = 1.0
        # lab[0,1] = -1.0
        # lab[1,2] = 1.0
        # lab[2,0] = -1.0
        tau = np.linalg.inv(np.column_stack([bu, bv, bw]))
        u = np.dot(lab, tau)
        # print u
        self.U = u
        # print "U: ", u
        return True

    def orient(self, u=None, v=None, rot=0.0, goniometer=[0.0, 0.0, 0.0]):
        """
        Creates a rotation matrix based on the u, v vectors.
        It is a prerequisite for outputting the oriented reciprocal lattice.
        """
        if self.unitcell is None:
            return None
        else:
            b = self.unitcell.B
            binv = self.unitcell.Binv
        if u is None:
            u = self.u.copy()
        if v is None:
            v = self.v.copy()
        # b, binv = self.B, self.Binv
        bu, bv = normalise(np.dot(b, u)), normalise(np.dot(b, v))
        if (length(bu) < 1e-6) or (length(bv) < 1e-6):
            return False
        bw = normalise(np.cross(bu, bv))
        if length(bw) < 1e-6:
            return False
        bv = normalise(np.cross(bw, bu))
        u, v = normalise(np.dot(binv, bu)), normalise(np.dot(binv, bv))
        phi, chi, omega = goniometer[0], goniometer[1], goniometer[2]
        axis_phi, axis_chi, axis_omega = (
            normalise(u.copy()),
            normalise(v.copy()),
            normalise(np.cross(v, u)),
        )
        rmat = np.dot(np.dot(self.Binv, arb_rotation(axis_omega, rot - omega)), self.B)
        axis_phi = np.dot(rmat, axis_phi)
        axis_chi = np.dot(rmat, axis_chi)
        axis_omega = np.dot(rmat, axis_omega)
        u = np.dot(rmat, u)
        v = np.dot(rmat, v)
        self.Rspace_Rmatrix = copy.deepcopy(rmat)
        rmat = np.dot(np.dot(self.Binv, arb_rotation(axis_chi, chi)), self.B)
        axis_phi = np.dot(rmat, axis_phi)
        axis_chi = np.dot(rmat, axis_chi)
        axis_omega = np.dot(rmat, axis_omega)
        u = np.dot(rmat, u)
        v = np.dot(rmat, v)
        self.Rspace_Rmatrix = np.dot(self.Rspace_Rmatrix, rmat)
        rmat = np.dot(np.dot(self.Binv, arb_rotation(axis_phi, phi)), self.B)
        axis_chi = np.dot(rmat, axis_chi)
        axis_omega = np.dot(rmat, axis_omega)
        axis_omega = np.dot(rmat, axis_omega)
        self.u = normalise(np.dot(rmat, u))
        self.v = normalise(np.dot(rmat, v))
        self.Rspace_Rmatrix = np.dot(self.Rspace_Rmatrix, rmat)
        # u = np.dot(rmat, u)
        # v = np.dot(rmat, v)
        # axis = np.cross(v, u)
        # rmat = np.dot(np.dot(self.Binv, arb_rotation(axis, rot)), self.B)
        # self.u = normalise(np.dot(rmat, u))
        # self.v = normalise(np.dot(rmat, v))
        itworked = self.makeU(self.u, self.v)
        self.rotmat = self.U
        self.themat = np.array(np.matrix(self.rotmat) * np.matrix(self.B))
        # self.themat = np.row_stack([themat[1],themat[0],themat[2]])
        return itworked


class SampleUnitcell:
    def __init__(
        self, abc=np.array([1.0, 1.0, 1.0]), angles=np.array([90.0, 90.0, 90.0])
    ):
        self.abc = abc
        self.angles = np.radians(angles)
        self.G = []
        self.Gstar = []
        self.B = []
        self.Binv = []
        self.repang = None
        self.repvec = None
        self.rotmat = None
        self.reciprocal_vectors()
        self.themat = []

    def makeG(self):
        g = np.zeros((3, 3))
        g[0, 0] = self.abc[0] ** 2
        g[1, 1] = self.abc[1] ** 2
        g[2, 2] = self.abc[2] ** 2
        g[0, 1] = self.abc[0] * self.abc[1] * np.cos(self.angles[2])
        g[0, 2] = self.abc[0] * self.abc[2] * np.cos(self.angles[1])
        g[1, 2] = self.abc[1] * self.abc[2] * np.cos(self.angles[0])
        g[1, 0] = g[0, 1]
        g[2, 0] = g[0, 2]
        g[2, 1] = g[1, 2]
        self.G = g

    def makeGstar(self):
        g = self.G
        gstar = np.linalg.inv(g)
        self.Gstar = gstar

    def makeRepvec(self):
        vecs = np.zeros(3)
        angs = np.zeros(3)
        vecs[0] = np.sqrt(self.Gstar[0, 0])
        vecs[1] = np.sqrt(self.Gstar[1, 1])
        vecs[2] = np.sqrt(self.Gstar[2, 2])
        angs[0] = np.arccos(self.Gstar[1, 2] / vecs[1] / vecs[2])
        angs[1] = np.arccos(self.Gstar[0, 2] / vecs[0] / vecs[2])
        angs[2] = np.arccos(self.Gstar[0, 1] / vecs[0] / vecs[1])
        self.repvec = vecs
        self.repang = angs

    def reciprocal_vectors(self):
        self.makeG()
        self.makeGstar()
        self.makeRepvec()
        self.makeB()
        # print self.repvec
        # print self.repang

    def makeB(self):
        vecs = self.repvec
        angs = self.repang
        b = np.zeros((3, 3))
        b[0, 0] = vecs[0]
        b[0, 1] = vecs[1] * np.cos(angs[2])
        b[0, 2] = vecs[2] * np.cos(angs[1])
        b[1, 1] = vecs[1] * np.sin(angs[2])
        b[1, 2] = -vecs[2] * np.sin(angs[1]) * np.cos(angs[0])
        b[2, 2] = 1 / self.abc[2]
        # b *= 2*np.pi
        binv = np.linalg.inv(b)
        self.B = b
        self.Binv = binv
        # print "B: ", b
        # print "B':", binv


class SampleLayer:
    def __init__(self):
        self.thickness = 0.0
        self.type = "unknown"
        self.composition = SampleComposition()
        self.unitcell = SampleUnitcell()
        self.orientation = SampleOrientation()


class Sample(DataNode):
    """
    Sample contains information about the sample
    composition, thickness, crystal unit cell and
    orientation. It should be assigned to an area
    on the SampleHolder. Then, the MeasPoints
    on the sample can follow in the tree structure.
    """

    def __init__(self, placetag=""):
        super().__init__(placetag=placetag)
        self.type = "Sample"
        self.metadata = {}
        self.layers = []
        self.nlayers = 0
        self.thickness = 0.0
        self.orientation = np.array([0.0, 0.0, 0.0])
        self.filename = None

    def extra_data(self):
        tdict = {}
        try:
            tdict["formula"] = self.composition.formula
        except:
            tdict["formula"] = ""
        try:
            tdict["sampletype"] = self.sampletype
        except:
            tdict["sampletype"] = "unknown"
        tdict["filename"] = self.filename
        return tdict

    def setFilename(self, fname):
        self.filename = fname

    def addLayer(self, nlayer):
        self.layers.append(nlayer)
        self.update()

    def removeLayer(self, number):
        try:
            self.layers.pop(number)
        except:
            return False
        else:
            self.update()
            return True

    def update(self):
        self.nlayers = len(self.layers)
        self.thickness = 0.0
        for l in self.layers:
            self.thickness += l.thickness

    def thicknessProfile(self):
        pass

    def currentUB(self):
        toplayer = self.layers[-1]
        orientation = self.orientation
        toplayer.orientation.assignUnitcell(toplayer.unitcell)
        rc = toplayer.orientation.orient(goniometer=orientation)
        if rc:
            return toplayer.orienation.themat

    def setSampletype(self, stype):
        self.sampletype = stype


class MeasPoint(DataNode):
    """
    Measurement Point is a logical representation of a single group of
    experimental setpoints, which may contain many data files,
    all of them measured (at least nominally) in the same conditions.
    Typically an experiment will contain multiple measurement points,
    and they, in turn, will contain multiple data files.
    """

    def __init__(self, placetag=""):
        super().__init__(placetag=placetag)
        self.type = "Measurement"
        self.comp_files = []
        self.metadata = {}

    def extra_data(self):
        tdict = {}
        return tdict

    def addFile(self, newFile):
        self.comp_files.append(newFile)

    def deleteFile(self, index):
        self.comp_files.pop(index)

    def returnYvalues(self):
        nfiles = len(self.child_ptrs)
        retvals = np.zeros((nfiles, 12))
        for n, ch in enumerate(self.child_ptrs):
            tdict = ch.extra_data()
            tline = []
            for kk in [
                "FIT_fwhm",
                "FIT_area",
                "FIT_centre",
                "ENERGY_fwhm",
                "ENERGY_area",
                "ENERGY_centre",
            ]:
                try:
                    tval = tdict[kk]
                except:
                    tline += [-1.0, -1.0]
                else:
                    try:
                        vals = [float(xx) for xx in tval.split(",")]
                    except:
                        vals = [-1.0, -1.0]
                    else:
                        if len(vals) == 2:
                            tline += [vals[0], vals[1]]
                        else:
                            tline += [-1.0, -1.0]
            # oneline = np.concatenate([tdict['FIT_centre'],  tdict['FIT_fwhm'] ,  tdict['FIT_area'] ,
            #                           tdict['ENERGY_centre'],  tdict['ENERGY_fwhm'], tdict['ENERGY_area'] ])
            retvals[n, :] = np.array(tline)
        return retvals

    def returnXvalues(self):
        retvals = {}
        heads = []
        keys = {}
        total = len(self.child_ptrs)
        for n, ch in enumerate(self.child_ptrs):
            heads.append(ch.getHeader())
        for n, thead in enumerate(heads):
            for kk in thead.keys():
                if kk in keys.keys():
                    keys[kk] += 1
                else:
                    keys[kk] = 1
        for kk in keys.keys():
            if keys[kk] == total:
                retvals[kk] = []
                for thead in heads:
                    retvals[kk].append(thead[kk])
        return retvals

    def returnCurves(self):
        curves = []
        for n, ch in enumerate(self.child_ptrs):
            temp = ch.getCurve()
            if temp is not None:
                curves.append((ch.name, temp))
        return curves

    def returnEnergyCurves(self):
        curves = []
        for n, ch in enumerate(self.child_ptrs):
            temp = ch.getEnergyCurve()
            if temp is not None:
                curves.append((ch.name, temp))
        return curves


class SingleFile(DataNode):
    """
    This is a metadata object that will accompany
    a single SIF file produced by the PEAXIS instrument.
    The information stored here will determine how a
    1D profile is extracted from the 2D data.
    The role of the file in the experiment will be
    determined by its position in the experiment tree.
    """

    def __init__(self, placetag=""):
        super().__init__(placetag=placetag)
        self.type = "SingleFile"
        self.filename = None
        self.fpath = ""
        self.data, self.header, self.logvals, self.logvalnames = None, None, None, None
        self.processing_history = []
        self.np_arrays = []
        self.np_profiles = []
        self.ADLER_params = {}
        self.create_ADLER_params()

    def create_ADLER_params(self):
        tdict = {}
        tdict["bpp"] = 960.78154
        tdict["tdb"] = 1
        tdict["cray"] = 3
        tdict["ffts"] = -1e9 * np.ones(2)
        tdict["poly"] = -100.0 * np.ones(3)
        tdict["eline"] = np.array([0, 2048])
        tdict["detlim"] = np.array([0, 2048])
        tdict["bkg_perc"] = 75
        tdict["segsize"] = 16
        tdict["eVpair1"] = -1 * np.ones(2)
        tdict["eVpair2"] = -1 * np.ones(2)
        tdict["redfac"] = 1.0
        # for the output, we always keep the value and the error
        tdict["FIT_centre"] = -1 * np.ones(2)
        tdict["FIT_fwhm"] = -1 * np.ones(2)
        tdict["FIT_area"] = -1.0 * np.ones(2)
        tdict["ENERGY_centre"] = -1.0 * np.ones(2)
        tdict["ENERGY_fwhm"] = -1.0 * np.ones(2)
        tdict["ENERGY_area"] = -1.0 * np.ones(2)
        for kk in tdict.keys():
            if kk not in self.ADLER_params.keys():
                self.ADLER_params[kk] = tdict[kk]

    def extra_data(self):
        tdict = {}
        tdict["filename"] = self.filename
        tdict["filepath"] = self.fpath
        for kk in self.ADLER_params.keys():
            try:
                len(self.ADLER_params[kk])
            except:
                tdict[kk] = self.ADLER_params[kk]
            else:
                tdict[kk] = ", ".join([str(x) for x in self.ADLER_params[kk]])
        return tdict

    def take_other_vars(self, tdict):
        for kk in tdict.keys():
            if str(kk) in ["filename"]:
                self.setFilename(tdict[kk])
                self.name = ".".join(str(tdict[kk]).split(".")[:-1])
            elif str(kk) in ["filepath"]:
                self.setPath(tdict[kk])
            else:
                try:
                    tfloat = float(tdict[kk])
                except:
                    try:
                        toks = tdict[kk].strip("[]() \n").split(",")
                    except:
                        self.ADLER_params[kk] = tdict[kk]
                    else:
                        if len(tdict[kk].strip("[]() \n").split(",")) > 1:
                            tlist = []
                            for tt in toks:
                                tlist.append(float(tt))
                            if kk in self.ADLER_params.keys():
                                self.ADLER_params[kk] = np.array(tlist)
                else:
                    self.ADLER_params[kk] = tfloat

    def setFilename(self, fname):
        self.filename = fname
        self.name = ".".join(str(fname).split(".")[:-1])

    def setPath(self, fpath):
        self.fpath = fpath.replace("\\", "/").replace("//", "/")

    def setHeader(self, thead):
        self.header = thead

    def setLogvals(self, tlog):
        self.logvals = tlog
        tnames = [str(x) for x in tlog.keys()]
        self.logvalnames = tnames

    def restoreValues(self):
        #        if len(self.header) == 0:
        #            self.header = None
        #        if len(self.logvals) == 0:
        #            self.logvals = None
        #        if len(self.logvalnames) == 0:
        #            self.logvalnames = None
        if self.header is None or self.logvals is None or self.logvalnames is None:
            thead, tlog = header_read(self.fpath)
            self.setHeader(thead)
            self.setLogvals(tlog)

    def getHeader(self):
        if self.header is None:
            self.restoreValues()
        return self.header

    def getLogvals(self):
        if self.logvals is None:
            self.restoreValues()
        return self.logvals

    def getLogvalNames(self):
        if self.logvalnames is None:
            self.restoreValues()
        return self.logvalnames

    def getCurve(self):
        tpath, tname = os.path.split(self.fpath)
        name_as_segments = simplify_number_range([tname])
        temp_name = "_".join(["Merged"] + name_as_segments)
        newname = temp_name + "_1D.txt"
        # newname = 'Merged_' + '.sif'.join(tname.split('.sif')[:-1]) +'_1D.txt'
        temppath = tpath + "/" + newname
        # temppath = temppath.replace('\\','/')
        try:
            obj = open(temppath, "r")
        except:
            print("Did not find ", temppath)
            return None
        else:
            obj.close()
            return temppath

    def getEnergyCurve(self):
        tpath, tname = os.path.split(self.fpath)
        name_as_segments = simplify_number_range([tname])
        temp_name = "_".join(["Merged"] + name_as_segments)
        newname = temp_name + "_1D_deltaE.txt"
        # newname = 'Merged_' + '.sif'.join(tname.split('.sif')[:-1]) +'_1D.txt'
        temppath = tpath + "/" + newname
        try:
            obj = open(temppath, "r")
        except:
            return None
        else:
            obj.close()
            return temppath

    def take_params(self, tempdict):
        super().take_params(tempdict)
        newdict = {}
        for kk in tempdict.keys():
            if str(kk) not in [
                "name",
                "type",
                "children",
                "metadata",
                "data",
                "place",
                "parent",
            ]:
                newdict[kk] = tempdict[kk]
        self.take_other_vars(newdict)
        # self.setFilename(tempdict['filename'])
        # self.setPath(tempdict['filepath'])


# rootnode = Experiment(rootdir = os.path.join('D:\\', 'Profile','jlb', 'testfiles',  'model'),
#                                   placetag = 'root')
## m1 = MeasPoint(placetag = 'meas1')
## f1 = SingleFile(placetag = 'file1')
## f2 = SingleFile(placetag = 'dud1')
## m1.setParent(rootnode)
## f1.setParent(m1)
## f2.setParent(m1)
# rootnode.SaveAll()

# here we start with the display part of the nodes
# This part allows us to show the nodes in a GraphicsView
# and to move them using Drag'n'Drop operations

node_types = {0: Experiment, 1: SampleHolder, 2: Sample, 3: MeasPoint, 4: SingleFile}

global_node_dictionary = {}
global_node_counter = 0


class TreeLayout:
    def __init__(self, headnode, scene):
        self.head = headnode
        self.current_head = headnode
        self.nsize = headnode.size
        self.scene = scene
        self.offset = (5.0, 5.0)
        self.positionItems()

    def positionItems(self):
        self.scene.removeItem(self.head)
        self.col = 0
        self.row = 0
        self.addNode(self.current_head)
        self.scene.update()

    def addNode(self, node, plusrow=0, pluscol=0):
        node.layout = self
        node.newPosition(
            (
                self.offset[0] + (self.col + pluscol) * self.nsize[0],
                self.offset[1] + (self.row + plusrow) * self.nsize[1],
            )
        )
        self.scene.addItem(node)
        self.row += 1
        for ch in node.children:
            self.addNode(ch, plusrow=plusrow, pluscol=ch.nesting())

    def clear_all(self):
        for i in self.current_head.children:
            i.destroy()

    def showChildren(self, dnode, nnode):
        temp = Node(nnode, placetag=dnode.placetag, innode=dnode)
        self.addNode(temp)
        for ch in dnode.child_ptrs:
            self.showChildren(ch, temp)

    def load(self, fullpath):
        fpath, fname = os.path.split(fullpath)
        self.current_head.innernode.rootdir = fpath
        self.current_head.innernode.restore(fname)
        for ch in self.current_head.innernode.child_ptrs:
            self.showChildren(ch, self.current_head)
        self.positionItems()


class Node(QGraphicsItem):
    def __init__(
        self, parent, coords=[0.0, 0.0], size=[90.0, 20.0], placetag="", innode=None
    ):
        super().__init__(parent)
        self.coords = coords
        self.size = size
        self.children = []
        self.parent = parent
        self.cfont = None
        self.layout = None
        self.betterrect = None
        self.dragStartPosition = QPointF(self.coords[0], self.coords[1])
        self.setAcceptDrops(True)
        self.colour = QCC.LightGray
        if parent is not None:
            parent.children.append(self)
        global global_node_counter
        global global_node_dictionary
        global last_object
        global_node_counter += 1
        global_node_dictionary["Node" + str(global_node_counter)] = self
        self.dictkey = "Node" + str(global_node_counter)
        self.fixed_nest = self.nesting()
        if innode is not None:
            self.innernode = innode
        else:
            self.innernode = node_types[self.fixed_nest](placetag=placetag)
        self.innernode.assignPlotNode(self)
        if parent is not None:
            # self.innernode.setParent(parent.innernode)
            parent.innernode.addChild(self.innernode)
        self.updColour()
        last_object += 1
        # self.setPos()

    def setFont(self, cfont):
        self.cfont = cfont

    def destroy(self):
        if self.fixed_nest > 0:
            for ch in self.children:
                ch.destroy()
            self.children = []
            self.innernode.parent_ptr.deleteChild(self.innernode)
            self.innernode = None
            global global_node_dictionary
            global_node_dictionary.pop(self.dictkey)
            if self.parent is not None:
                if self in self.parent.children:
                    index = self.parent.children.index(self)
                    self.parent.children.pop(index)

    def removeChild(self, child):
        if child in self.children:
            self.innernode.deleteChild(child.innernode)
            ind = self.children.index(child)
            self.children.pop(ind)

    def newChild(self, child):
        if child.fixed_nest == (self.fixed_nest + 1):
            if child not in self.children:
                child.parent.removeChild(child)
                self.children.append(child)
                child.parent = self
                self.innernode.addChild(child.innernode)

    def newPosition(self, coords):
        self.coords = coords

    def nesting(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.nesting() + 1

    def boundingRect(self):
        if self.betterrect is not None:
            return self.betterrect
        else:
            return QRectF(self.coords[0], self.coords[1], self.size[0], self.size[1])

    def updColour(self):
        if self.fixed_nest == 0:
            self.colour = QCC.Svg.gold
        elif self.fixed_nest == 1:
            self.colour = QCC.Svg.darksalmon
        elif self.fixed_nest == 2:
            self.colour = QCC.Svg.darkorange
        elif self.fixed_nest == 3:
            self.colour = QCC.Svg.chartreuse
        elif self.fixed_nest == 4:
            self.colour = QCC.Svg.mediumspringgreen
        else:
            self.colour = QCC.LightGray

    def paint(self, painter, option, widget):
        #        if self.innernode is not None:
        #            # startrect = QRectF(self.coords[0], self.coords[1],
        #            #                   self.size[0], self.size[1])
        #            labtext = self.innernode.name
        #            cfont = self.layout.scene.font()
        #            fmetrics = QFontMetricsF(cfont)
        #            newrect = fmetrics.boundingRect(labtext)#.translated(self.coords[0],self.coords[1])
        #            # lastrect = startrect.united(newrect)
        #            # temprect = QRectF(newrect.x(), newrect.y(),  newrect.width(), newrect.height())
        #            self.size[0] = newrect.width()+10
        #            self.size[1] = newrect.height()
        if self.fixed_nest == 0:
            painter.setPen(QPen(QCC.Black, 1))
            painter.setBrush(QBrush(self.colour))
            painter.drawRect(self.coords[0], self.coords[1], self.size[0], self.size[1])
        elif self.fixed_nest == 1:
            painter.setPen(QPen(QCC.Black, 1))
            painter.setBrush(QBrush(self.colour))
            painter.drawEllipse(
                self.coords[0], self.coords[1], self.size[0], self.size[1]
            )
        elif self.fixed_nest == 2:
            painter.setPen(QPen(QCC.Black, 1))
            painter.setBrush(QBrush(self.colour))
            painter.drawRect(self.coords[0], self.coords[1], self.size[0], self.size[1])
        elif self.fixed_nest == 3:
            painter.setPen(QPen(QCC.Black, 1))
            painter.setBrush(QBrush(self.colour))
            painter.drawEllipse(
                self.coords[0], self.coords[1], self.size[0], self.size[1]
            )
        elif self.fixed_nest == 4:
            painter.setPen(QPen(QCC.Black, 1))
            painter.setBrush(QBrush(self.colour))
            painter.drawRect(self.coords[0], self.coords[1], self.size[0], self.size[1])
        else:
            painter.setPen(QPen(QCC.White, 2))
            painter.setBrush(QBrush(self.colour))
            painter.drawEllipse(
                self.coords[0], self.coords[1], self.size[0], self.size[1]
            )
        painter.drawLine(
            self.coords[0],
            self.coords[1] + self.size[1] / 2,
            self.coords[0] - self.size[0] / 2,
            self.coords[1] + self.size[1] / 2,
        )
        painter.drawLine(
            self.coords[0] - self.size[0] / 2,
            self.coords[1],
            self.coords[0] - self.size[0] / 2,
            self.coords[1] + self.size[1] / 2,
        )
        if self.innernode is not None:
            painter.drawText(
                self.coords[0] + 5,
                self.coords[1] + self.size[1] - 2,
                self.innernode.name,
            )

    def mousePressEvent(self, event):
        # print(event.button())
        if Qt.MouseEventFlag.MouseEventCreatedDoubleClick & event.flags():
            return None
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragStartPosition = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self.dragStartPosition = event.pos()
        #

    def mouseMoveEvent(self, event):
        # print(event.button())
        if not event.button() == Qt.MouseButton.RightButton:
            # print((event.pos() - self.dragStartPosition).manhattanLength() )
            if (
                event.pos() - self.dragStartPosition
            ).manhattanLength() >= QApplication.startDragDistance():
                print("The Node is being dragged!")
                drag = QDrag(event.widget())
                mime = QMimeData()
                mime.setText(self.dictkey)
                drag.setMimeData(mime)
                drag.exec()

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)

    def dragEnterEvent(self, event):
        self.colour = QCC.Yellow
        self.update()

    def dragLeaveEvent(self, event):
        self.updColour()
        self.update()

    def dropEvent(self, event):
        self.updColour()
        self.update()
        global global_node_dictionary
        global last_object
        tempmime = event.mimeData()
        # print('dropEvent source: ', event.source())
        # print('dropEvent source identifier: ', event.source().whoami())
        try:
            source_id = event.source().whoami()
        except AttributeError:
            if tempmime.hasText():
                key = event.mimeData().text()
                source = global_node_dictionary[key]
                print(source)
                self.newChild(source)
                self.layout.positionItems()
        else:
            if "Files" in source_id and "TableView" in source_id:
                spos = event.source().dragStartPosition
                index = event.source().rowAt(spos.y())
                # pos = QPoint(0, index)
                index2 = event.source().indexAt(spos)
                # print('Vertical position:', spos.y(), ', table row:',  index, ', model index:', index2.column(),  index2.row())
                name = event.source().model().itemFromIndex(index2).text()
                fpath = event.source().model()._file_dict[name]
                thead = event.source().model()._header_dict[name]
                tlog = event.source().model()._log_dict[name]
                print(name)
                if (
                    self.fixed_nest == 3
                ):  # here we add a new node, which is a SingleFile
                    # global last_object
                    temp = Node(
                        self, [80.0, 20.0], placetag="Object" + str(last_object)
                    )
                    temp.innernode.setFilename(name)
                    temp.innernode.setPath(fpath)
                    temp.innernode.setHeader(thead)
                    temp.innernode.setLogvals(tlog)
                elif (
                    self.fixed_nest == 4
                ):  # here we change the file name inside an existing SingleFile
                    self.innernode.setFilename(name)
                    self.innernode.setPath(fpath)
                    self.innernode.setHeader(thead)
                    self.innernode.setLogvals(tlog)
                self.layout.positionItems()
            elif "Samples" in source_id and "IconWidget" in source_id:
                samp_object = event.source().current_item
                sample_file = samp_object.filename
                sample_type = samp_object.sampletype
                if self.fixed_nest == 1:  # here we add a new Sample node
                    # global last_object
                    temp = Node(
                        self, [80.0, 20.0], placetag="Object" + str(last_object)
                    )
                    temp.innernode.setFilename(sample_file)
                    temp.innernode.setSampletype(sample_type)
                elif self.fixed_nest == 2:  # here we overwrite the sample definition
                    self.innernode.setFilename(sample_file)
                    self.innernode.setSampletype(sample_type)
                self.layout.positionItems()


class TableView(QTableView):
    release_items = pyqtSignal(object)

    def __init__(self, parent):
        super().__init__(parent)

    def mousePressEvent(self, event):
        # print(event.button())
        if Qt.MouseEventFlag.MouseEventCreatedDoubleClick & event.flags():
            return None
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragStartPosition = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self.dragStartPosition = event.pos()
        #

    def mouseMoveEvent(self, event):
        # print(event.button())
        if not event.button() == Qt.MouseButton.RightButton:
            # print((event.pos() - self.dragStartPosition).manhattanLength() )
            if (
                event.pos() - self.dragStartPosition
            ).manhattanLength() >= QApplication.startDragDistance():
                print("The Node is being dragged!")
                print("Position:", event.pos())
                drag = QDrag(self)
                mime = QMimeData()
                # mime.setText(self.dictkey)
                drag.setMimeData(mime)
                drag.exec()

    def whoami(self):
        return "Files:TableView"

    @pyqtSlot()
    def triggerList(self):
        selmod = self.selectionModel()
        rows = selmod.selectedRows()
        flist = []
        for rrr in rows:
            fname = rrr.data()
            flist.append(fname)
        self.release_items.emit(flist)


class IconWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(200, 200)
        self.setFrameStyle(QFrame.Sunken | QFrame.StyledPanel)
        self.setAcceptDrops(True)
        self.icons = []
        self.objects = []
        self.pixmaps = {}
        for kk in ["amorphous", "powder", "crystal", "pellet", "unknown"]:
            self.pixmaps[kk] = QPixmap(resource_path(kk + ".png"))
        self.current_item = None
        dummycrystal = DummySample()
        dummycrystal.sampletype = "crystal"
        dummypowder = DummySample()
        dummypowder.sampletype = "powder"
        self.newObject(dummycrystal)
        self.newObject(dummypowder)

    def getPixmap(self, sampletype="unknown"):
        csize = self.size()
        totalx, totaly = csize.width(), csize.height()
        if totalx < 512 or totaly < 512:
            iconsize = 64
        if sampletype in self.pixmaps.keys():
            return self.pixmaps[sampletype].scaled(iconsize, iconsize)
        else:
            return self.pixmaps["unknown"].scaled(iconsize, iconsize)

    def nextFreePosition(self):
        csize = self.size()
        totalx, totaly = csize.width(), csize.height()
        npics = len(self.icons)
        # ratio = totalx/totaly
        perline = int(round(npics**0.5)) + 1
        stepsize = (totalx - 20) / perline
        y = 10 + stepsize * int(npics / perline)
        x = 10 + stepsize * int(npics % perline)
        return x, y

    def newObject(self, something):
        temp = QLabel(self)
        npos = self.nextFreePosition()
        self.icons.append(temp)
        self.objects.append(something)
        temp.setPixmap(self.getPixmap(something.sampletype))
        temp.move(npos[0], npos[1])
        temp.show()
        temp.setAttribute(Qt.WA_DeleteOnClose)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-dnditemdata"):
            if event.source() == self:
                event.setDropAction(Qt.MoveAction)
                event.accept()
            else:
                event.acceptProposedAction()
        else:
            event.ignore()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-dnditemdata"):
            itemData = event.mimeData().data("application/x-dnditemdata")
            dataStream = QDataStream(itemData, QIODevice.ReadOnly)

            pixmap = QPixmap()
            offset = QPoint()
            dataStream >> pixmap >> offset

            newIcon = QLabel(self)
            newIcon.setPixmap(pixmap)
            newIcon.move(event.pos() - offset)
            newIcon.show()
            newIcon.setAttribute(Qt.WA_DeleteOnClose)

            if event.source() == self:
                event.setDropAction(Qt.MoveAction)
                event.accept()
            else:
                event.acceptProposedAction()
        else:
            event.ignore()

    def mousePressEvent(self, event):
        child = self.childAt(event.pos())
        if not child:
            return
        else:
            index = self.icons.index(child)
            self.current_item = self.objects[index]
        pixmap = QPixmap(child.pixmap())

        itemData = QByteArray()
        dataStream = QDataStream(itemData, QIODevice.WriteOnly)
        dataStream << pixmap << QPoint(event.pos() - child.pos())

        mimeData = QMimeData()
        mimeData.setData("application/x-dnditemdata", itemData)

        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos() - child.pos())

        tempPixmap = QPixmap(pixmap)
        painter = QPainter()
        painter.begin(tempPixmap)
        painter.fillRect(pixmap.rect(), QColor(127, 127, 127, 127))
        painter.end()

        child.setPixmap(tempPixmap)

        if drag.exec_(Qt.CopyAction | Qt.MoveAction, Qt.CopyAction) == Qt.MoveAction:
            child.close()
        else:
            child.show()
            child.setPixmap(pixmap)

    def whoami(self):
        return "Samples:IconWidget"


class DummySample:
    def __init__(self, filename=""):
        self.filename = filename
        self.sampletype = "unknown"
