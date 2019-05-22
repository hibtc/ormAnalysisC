import numpy as np
from cpymad.madx import Madx
from yaml import safe_load
import matplotlib.pyplot as plt

class OrmAnalysis:

    """
    This Class intends to compute the measured and the modeled orbit
    response matrix and is able to plot the data for control of the
    data quality.
    @param dataFile is the file path where the measured data is. It is
           expected to be a measurement as produced by madgui in yaml format.
    @param madxModelFile is the file path to the MAD-X model file. The model
           should run in MAD-X.
    """
    def __init__(self, dataFile, madxModelFile):

        self.madxModelFile    = madxModelFile
        self.dataFile       = dataFile
        self.data           = self.readData(dataFile)

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

    def computeOrmMeasured(self, monitor, kickers, kicks, data2 = 0):
        """
        Computes the measured orbit responses at an specifical
        monitor
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
                dataBeam.append(shot[monitor])

            mean = np.mean(dataBeam, axis = 0)
            stdDev = np.std(dataBeam, axis = 0)

            beamMess.append(np.array(mean[:2]))
            beamMessDev.append(np.array(stdDev[:2]))

        ORMcxx = []
        dORMcxx = []
        # Note that len(kicks) + 1 == len(beamMess) must hold
        # this condition could be implemented as control
        for k in range( len(kicks) ):
            k0 = madx.globals[kickers[k]]
            kickDiff = (kicks[k]-k0)
            cxx  = (beamMess[k+1]-beamMess[0])/kickDiff
            # Gaussian error propagation
            dcxx = np.sqrt(beamMessDev[k+1]**2 + beamMessDev[0]**2)/kickDiff
            ORMcxx.append(cxx)
            dORMcxx.append(dcxx)

        madx.input('STOP;')

        ORMcxx = np.transpose(ORMcxx)
        dORMcxx = np.transpose(dORMcxx)

        if data2 != 0: self.data = data1

        return ORMcxx, dORMcxx

    def ORMModel(self, monitor, kickers,
                 kick = 2e-4 ):
        """
        Computes the orbit response according to the MADX model
        with the same parameters as for a given measurement and
        for a given monitor and kicker list
        @param monitor(String) is the name of the monitor for which the
               orbit response is going to be computed
        @param kickers(String list) is a list containing the used kickers
               in the measurement
        """
        # This are initial conditions for the Twiss Module of MAD-X
        # it doesn't see to be a strong dependence on them
        dx = 1.0e-4
        dpx = 1.0e-6
        dy = 1.0e-4
        dpy = 1.0e-6

        madx = Madx(stdout=False)
        madx.call(file=self.madxModelFile, chdir=True)
        elems = madx.sequence.hht3.expanded_elements
        iMonitor = elems.index(monitor)

        Cxx = []
        Cxy = []

        for k in kickers:

            madx.globals.update( self.data['model'] )
            twiss1 = madx.twiss(sequence='hht3', RMatrix=True,
                                alfx = -7.2739282, alfy = -0.23560719,
                                betx = 44.404075,  bety = 3.8548009,
                                x = dx, y = dy,
                                px = dpx, py = dpy)

            # This should be hopefully zero
            y1 = twiss1.y[iMonitor]
            x1 = twiss1.x[iMonitor]

            madx.globals[k] += kick

            twiss2 = madx.twiss(sequence='hht3', RMatrix=True,
                                alfx = -7.2739282, alfy = -0.23560719,
                                betx = 44.404075,  bety = 3.8548009,
                                x = dx, y = dy,
                                px = dpx, py = dpy)

            y2 = twiss2.y[iMonitor]
            x2 = twiss2.x[iMonitor]
            # Horizontal and vertical response
            cx = ( x2 - x1 ) / kick
            cy = ( y2 - y1 ) / kick

            Cxx.append(cx)
            Cxy.append(cy)

        madx.input('STOP;')
        return Cxx, Cxy

    def computeORM(self, messFile2 = '', plot=True, saveFigs=False):
        """
        This functions plots the data of a given file, together
        with the modeled (expected) orbit response from MADX.
        If a second measurement file is given, it will be assumed the
        same conditions hold (same Monitor, same Kickers).
        @param messFile2 is the second measurement, if the monitors
               of both measurements don't agree, nothing is plotted
        @param plot(boolean) plots the data and displays it
        @param saveFigs(boolean) saves the plots if True
        """
        monitor = self.getMonitor()
        kickers, kicks = self.getKicks()
        ormG, dormG   = self.computeOrmMeasured(monitor, kickers, kicks)
        ormMx, ormMy  = self.ORMModel(monitor, kickers)

        y1  = ormG[0]
        dy1 = dormG[0]

        y2  = ormG[1]
        dy2 = dormG[1]

        if messFile2 != '':
            data2 = self.readData(messFile2)
            ormG2, dormG2   = self.computeOrmMeasured(monitor, kickers,
                                                      kicks, data2)
            y12  = ormG2[0]
            dy12 = dormG2[0]

            y22  = ormG2[1]
            dy22 = dormG2[1]

        if(plot):
            self.plotData(monitor, kickers, y1, dy1, y2, dy2, ormMx, ormMy)
            if(messFile2!=''):
                self.plotData(monitor, kickers,
                              y12, dy12, y22, dy22,
                              ormMx, ormMy,plotModel=False)

            plt.show()
            plt.clf()
            plt.cla()
            plt.close()

        Cxx = ( ormMx - y1 ) / dy1
        Cyy = ( ormMy - y2 ) / dy2

        return Cxx, Cyy

    def plotData(self, monitor, kickers,
                 y1, dy1, y2, dy2, ormMx, ormMy,
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
        plt.xticks(x, kickers, rotation='vertical')
        plt.title("Monitor: {}".format(monitor))
        plt.legend(loc=0)

        if (save):
            plt.savefig('Results/{}h'.format(monitor))
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
        plt.xticks(x, kickers, rotation='vertical')
        plt.title("Monitor: {}".format(monitor))
        plt.legend(loc=0)

        if (save):
            plt.savefig('Results/{}v'.format(monitor))
            plt.close()
