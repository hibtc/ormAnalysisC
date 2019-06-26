import numpy as np
from cpymad.madx import Madx
from yaml import safe_load
import matplotlib.pyplot as plt

class OrmAnalysis:

    """
    This Class intends to compute the measured and the modeled orbit
    response matrix and is able to plot the data for control of the
    data quality.
    It is also able to compute the derivative of the ORM for a given
    list of parameters.
    @param dataFile is the file path where the measured data is. It is
           expected to be a measurement as produced by madgui in yaml format.
    @param madxModelFile is the file path to the MAD-X model file. The model
        should run in MAD-X.
    """
    def __init__(self, dataFile, madxModelFile):

        self.madxModelFile = madxModelFile
        self.dataFile      = dataFile
        self.data          = self.readData(dataFile)
        self.monitor       = self.getMonitor()
        self.kickers, self.kicks = self.getKicks()
        # This are initial conditions for the Twiss Module of MAD-X
        # it doesn't seem to be a strong dependence on them
        self.dx = 1.0e-4
        self.dpx = 1.0e-6
        self.dy = 1.0e-4
        self.dpy = 1.0e-6
        self.madx = Madx(stdout=False)
        self.madx.call(file=self.madxModelFile, chdir=True)

    def readData(self, dataFile):
        """
        Opens the data file including the measurements with YAML
        @param dataFile
        """
        with open(dataFile) as f:
            data = safe_load(f)

        # Normal Thomas ordering for kickers
        # namely according to its position s and in increasing order
        knobs = data['knobs']
        data['records'] = sorted( data['records'],
                                  key=lambda record: -1 if not record['optics']
                                  else knobs.index(
                                          list(record['optics'])[0]))
        return data

    def getMonitor(self):
        """
        Returns the name of the monitor at where it
        was measured
        """
        records = self.data['records']
        mess0 = records[0]['shots']
        monitor = list(mess0[0].keys())

        return monitor[0]

    def getKicks(self):
        """
        Returns a list of kickers names and corrector kick angles
        that were used in the measurement
        """
        records = self.data['records']
        kickers, kicks = [], []
        for messung in records:
            kicker = list(messung['optics'].keys())
            if ( len(kicker) != 0 ):
                kick = messung['optics']
                kickers.append(kicker[0])
                kicks.append(kick[kicker[0]])

        return kickers, kicks

    def computeOrmMeasured(self, data2 = 0):
        """
        Computes the measured orbit responses at an specifical
        monitor, returns the orbit response entries and their errors
        as two arrays. The first entry of the arrays are the horizontal
        response and the second is vertical response respectively.
        @param monitor(String) is the name of the monitor for which the
               orbit response is going to be computed
        @param kickers(String list) is a list containing the used kickers
               in the measurement
        @param kicks(double list) is a list containing the kick at the
               corrector kicker
        """
        if data2 != 0:
            data1 = self.data
            self.data = data2

        madx = Madx(stdout=False)
        madx.call(file=self.madxModelFile, chdir=True)
        madx.globals.update( self.data['model'] )

        records = self.data['records']
        # The measurements and their statistical errors are stored here
        beamMess = []
        beamMessDev = []

        for messung in records:
            shots = messung['shots']
            dataBeam = []
            for shot in shots:
                dataBeam.append(shot[self.monitor])

            mean = np.mean(dataBeam, axis = 0)
            stdDev = np.std(dataBeam, axis = 0)

            beamMess.append(np.array(mean[:2]))
            beamMessDev.append(np.array(stdDev[:2]))

        ORMcxx = []
        dORMcxx = []
        # Note that len(kicks) + 1 == len(beamMess) must hold
        # this condition could be implemented as control
        for k in range( len(self.kicks) ):
            k0 = madx.globals[self.kickers[k]]
            kickDiff = (self.kicks[k]-k0)
            cxx  = (beamMess[k+1]-beamMess[0])/kickDiff
            # Gaussian error propagation
            dcxx = np.sqrt(beamMessDev[k+1]**2 + beamMessDev[0]**2)/(kickDiff*np.sqrt(len(beamMessDev[k+1])))
            ORMcxx.append(cxx)
            dORMcxx.append(dcxx)

        madx.input('STOP;')

        ORMcxx = np.transpose(ORMcxx)
        dORMcxx = np.transpose(dORMcxx)

        if data2 != 0: self.data = data1

        return ORMcxx, dORMcxx

    def computeOrmModel(self, kick = 2e-4 ):
        """
        Computes the orbit response according to the MADX model
        with the same parameters as for a given measurement and
        for a given monitor and kicker list.
        """
        madx = self.madx
        elems = madx.sequence.hht3.expanded_elements
        iMonitor = elems.index(self.monitor)

        Cxx = []
        Cxy = []

        for k in self.kickers:
            madx.globals.update( self.data['model'] )
            twiss1 = madx.twiss(sequence='hht3', RMatrix=True,
                                alfx = -7.2739282, alfy = -0.23560719,
                                betx = 44.404075,  bety = 3.8548009,
                                x = self.dx, y = self.dy,
                                px = self.dpx, py = self.dpy)

            # This should be hopefully zero
            y1 = twiss1.y[iMonitor]
            x1 = twiss1.x[iMonitor]

            madx.globals[k] += kick

            twiss2 = madx.twiss(sequence='hht3', RMatrix=True,
                                alfx = -7.2739282, alfy = -0.23560719,
                                betx = 44.404075,  bety = 3.8548009,
                                x = self.dx, y = self.dy,
                                px = self.dpx, py = self.dpy)

            y2 = twiss2.y[iMonitor]
            x2 = twiss2.x[iMonitor]
            # Horizontal and vertical response
            cx = ( x2 - x1 ) / kick
            cy = ( y2 - y1 ) / kick

            Cxx.append(cx)
            Cxy.append(cy)

        madx.input('STOP;')
        return Cxx, Cxy

    def computeOrmModelpList(self, pList, p0List, kick = 2e-4):
        """
        Since there is no method overloading in Python
        I had to change the name of the method above,
        but it does the same, but changing the parameters
        in the model of MAD-X.
        """
        madx = self.madx
        elems = madx.sequence.hht3.expanded_elements
        iMonitor = elems.index(self.monitor)

        Cxx = []
        Cxy = []

        for k in self.kickers:

            madx.globals.update( self.data['model'] )
            """
            We update the model given the parameter list
            before we compute the Orbit Response
            """
            for p in range(len(pList)):
                madx.globals[pList[p]] += p0List[p]

            twiss1 = madx.twiss(sequence='hht3', RMatrix=True,
                                alfx = -7.2739282, alfy = -0.23560719,
                                betx = 44.404075,  bety = 3.8548009,
                                x = self.dx, y = self.dy,
                                px = self.dpx, py = self.dpy)

            # This should be hopefully zero
            y1 = twiss1.y[iMonitor]
            x1 = twiss1.x[iMonitor]

            madx.globals[k] += kick

            twiss2 = madx.twiss(sequence='hht3', RMatrix=True,
                                alfx = -7.2739282, alfy = -0.23560719,
                                betx = 44.404075,  bety = 3.8548009,
                                x = self.dx, y = self.dy,
                                px = self.dpx, py = self.dpy)

            y2 = twiss2.y[iMonitor]
            x2 = twiss2.x[iMonitor]
            # Horizontal and vertical response
            cx = ( x2 - x1 ) / kick
            cy = ( y2 - y1 ) / kick

            Cxx.append(cx)
            Cxy.append(cy)

        return np.array(Cxx), np.array(Cxy)

    def computeDorm(self, pList, p0List, kick = 2e-4, dp = 1e-9):
        """
        Computes the orbit response derivative for a given
        list of parameters.
        @param pList is the list of parameters the derivative should
               be computed to
        @param p0List is the list of the position around the derivative
               will be computed to
        @param dp0 cannot be smaller than 1e-9 for numerical precision
               reasons!!! Going smaller changes drastically the computation.
               Going up leaves the same results up to numerical precision.
        """
        madx = self.madx
        p0List = np.array(p0List)
        ormMx1, ormMy1 = self.computeOrmModelpList(pList, p0List)
        dCx = []
        dCy = []
        """
        I had to do this because lists in Python are mutable...
        I should ask Thomas for a better implementation
        """
        for p in range(len(pList)):
            pList1 = np.array(p0List)
            pList1[p] += dp
            #print(p)
            #print(p0List)
            #print(pList1)
            ormMx2, ormMy2 = self.computeOrmModelpList(pList, pList1)
            dCxdP = (ormMx2 - ormMx1)/dp
            dCydP = (ormMy2 - ormMy1)/dp
            dCx.append(dCxdP)
            dCy.append(dCydP)

        return np.array(dCx), np.array(dCy)

    def visualizeData(self, messFile2 = '', saveFigs=False):
        """
        This functions plots the data of a given file, together
        with the modeled (expected) orbit response from MADX.
        If a second measurement file is given, it will be assumed the
        same conditions hold (same Monitor, same Kickers).
        @param messFile2 is the second measurement
        @param saveFigs(boolean) saves the plots if True
        """
        ormG, dormG   = self.computeOrmMeasured()
        ormMx, ormMy  = self.computeOrmModel()

        # Horizontal response
        y1  = ormG[0]
        dy1 = dormG[0]
        # Vertical response
        y2  = ormG[1]
        dy2 = dormG[1]

        if messFile2 != '':
            data2 = self.readData(messFile2)
            ormG2, dormG2   = self.computeOrmMeasured(data2)
            y12  = ormG2[0]
            dy12 = dormG2[0]

            y22  = ormG2[1]
            dy22 = dormG2[1]

        self.plotData(y1, dy1, y2, dy2, ormMx, ormMy)
        if(messFile2!=''):
            self.plotData(y12, dy12, y22, dy22,
                          ormMx, ormMy,plotModel=False)
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def plotData(self, y1, dy1, y2, dy2, ormMx, ormMy,
                 save = False, plotModel = True):

        x = np.linspace(0,len(y1),len(y1))

        plt.figure(1,figsize=(8,8))
        plt.errorbar(x, y1, yerr=dy1, label="Measurement",
                     marker="x", linestyle="",capsize=5)
        if(plotModel):
            plt.plot(x, ormMx, label="Model",
                     marker=".",linestyle = "-")
        plt.xlabel('Kicker')
        plt.ylabel(r'Horizontal Orbit Response [mm mrad$^{-1}$]')
        locs, labels = plt.xticks()
        plt.xticks(x, self.kickers, rotation='vertical')
        plt.title("Monitor: {}".format(self.monitor))
        plt.legend(loc=0)

        if (save):
            plt.savefig('Results/{}h'.format(self.monitor))
            plt.close()

        plt.figure(2,figsize=(8,8))
        plt.errorbar(x, y2, yerr=dy2, label="Measurement",
                     marker="x", linestyle="",capsize=5)
        if(plotModel):
            plt.plot(x, ormMy, label="Model",
                     marker=".",linestyle = "-")
        plt.xlabel('Kicker')
        plt.ylabel(r'Vertical Orbit Response [mm mrad$^{-1}$]')
        locs, labels = plt.xticks()
        plt.xticks(x, self.kickers, rotation='vertical')
        plt.title("Monitor: {}".format(self.monitor))
        plt.legend(loc=0)

        if (save):
            plt.savefig('Results/{}v'.format(self.monitor))
            plt.close()
