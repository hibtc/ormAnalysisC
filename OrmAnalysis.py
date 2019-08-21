from cpymad.madx import Madx
from yaml import safe_load

import numpy as np
import matplotlib.pyplot as plt

class OrbitResponse:

    """
    Computes the measured and the modeled orbit response matrix at a 
    single monitor with the available kickers in the transfer line

    @param dataFile is the file path where the measured data is. It is
           expected to be a measurement as produced by madgui in yaml format.
    @param madxModelFile is the file path to the MAD-X model file. The model
           should run in MAD-X.
    """
    def __init__(self, dataFile, madxModelFile):
        self.madxModelFile       = madxModelFile
        self.dataFile            = dataFile
        self.data                = self.readData(dataFile)
        self.monitor             = self.getMonitor()
        self.kickers, self.kicks = self.getKicks()
        self.sequence            = self.getSequence()
        self.madx                = Madx(stdout=False)
        self.madx.call(file=self.madxModelFile, chdir=True)

        # This are the initial conditions for the Twiss Module of MAD-X
        # there doesn't seem to be a strong dependence on them
        self.dx = 1.0e-4
        self.dpx = 1.0e-6
        self.dy = 1.0e-4
        self.dpy = 1.0e-6

    def readData(self, dataFile):
        with open(dataFile) as f: data = safe_load(f)
        # Normal ordering for kickers.
        # Namely, according to their position s
        # and in increasing order
        knobs = data['knobs']
        data['records'] = sorted( data['records'],
                                  key=lambda record: -1 if not record['optics']
                                  else knobs.index(
                                          list(record['optics'])[0]))
        return data

    def setData(self, dataFile):
        self.dataFile = dataFile
        self.data     = self.readData(dataFile)
        self.monitor  = self.getMonitor()
        self.kickers, self.kicks = self.getKicks()

    def getMonitor(self):
        records = self.data['records']
        mess0   = records[0]['shots']
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

    def getSequence(self):
        sequences = ['hht1','hht2','hht3','hht4','hht5']
        try:
            for seqName in sequences:
                if seqName in self.madxModelFile:
                    return seqName
        except:
            print('Sequence not found!')
            
    def ormMeasured(self):
        """
        Computes the measured orbit responses at an specifical
        monitor, returns the orbit response entries and their errors
        as two arrays. The first entry of the arrays are the horizontal
        response and the second is vertical response respectively.
        """
        madx = self.madx
        self.madx.call(file=self.madxModelFile, chdir=True)
        self.madx.globals.update(self.data['model'])
        records     = self.data['records']
        beamMess    = []
        beamMessErr = []
        for messung in records:
            shots    = messung['shots']
            dataBeam = []
            for shot in shots: dataBeam.append(shot[self.monitor])
            mean   = np.mean(dataBeam, axis = 0)
            stdDev = np.std (dataBeam, axis = 0) / np.sqrt(len(dataBeam))
            # Last two entries correspond to the beam envelope measurement.
            # (That's why we just take the first two) 
            beamMess.append(np.array(mean[:2]))
            beamMessErr.append(np.array(stdDev[:2]))
        orbitResponse    = []
        orbitResponseErr = []
        # Note that len(kicks) + 1 == len(beamMess) must hold
        # this condition could be implemented as control
        for k in range(len(self.kicks)):
            k0       = madx.globals[self.kickers[k]]
            kickDiff = (self.kicks[k] - k0)
            # First measurement is always the reference measurement
            orbR_k    = (beamMess[k+1] - beamMess[0]) / kickDiff
            # Gaussian error propagation
            orbRErr_k = np.sqrt(beamMessErr[k+1]**2 + beamMessErr[0]**2) \
                        / kickDiff
            orbitResponse.append(orbR_k)
            orbitResponseErr.append(orbRErr_k)
        orbitResponse    = np.transpose(orbitResponse)
        orbitResponseErr = np.transpose(orbitResponseErr)
        return orbitResponse, orbitResponseErr

    def ormModel(self, kick=2e-4):
        """
        Computes the orbit response according to the MADX model
        with the same parameters as in the measurement
        """
        madx = self.madx
        madx.call(file=self.madxModelFile, chdir=True)
        elems    = madx.sequence[self.sequence].expanded_elements
        iMonitor = elems.index(self.monitor)

        madx.globals.update(self.data['model'])
        twiss0 = madx.twiss(sequence=self.sequence, RMatrix=True,
                            alfx = -7.2739282, alfy = -0.23560719,
                            betx = 44.404075,  bety = 3.8548009,
                            x = self.dx, y = self.dy,
                            px = self.dpx, py = self.dpy)
        x0 = twiss0.x[iMonitor]
        y0 = twiss0.y[iMonitor]
        orbitResponse_x = []
        orbitResponse_y = []
        for k in self.kickers:
            madx.globals.update(self.data['model'])
            madx.globals[k] += kick
            twiss1 = madx.twiss(sequence=self.sequence, RMatrix=True,
                                alfx = -7.2739282, alfy = -0.23560719,
                                betx = 44.404075,  bety = 3.8548009,
                                x = self.dx, y = self.dy,
                                px = self.dpx, py = self.dpy)
            x1 = twiss1.x[iMonitor]
            y1 = twiss1.y[iMonitor]
            # Horizontal and vertical response
            cx = (x1 - x0) / kick
            cy = (y1 - y0) / kick
            orbitResponse_x.append(cx)
            orbitResponse_y.append(cy)
        return orbitResponse_x, orbitResponse_y

    def ormModelpList(self, pList, dpList, kick=2e-4):
        """
        Computes the ORM and allows a user-defined list of
        parameters to be changed during the computation
        @param pList is a list containing the names of the
               parameters
        @param dpList is the absolute value by which the parameter
               has to be changed in the computation from the nominal
               value given in the measurement (self.data['model'])
        """
        madx     = self.madx
        elems    = madx.sequence[self.sequence].expanded_elements
        iMonitor = elems.index(self.monitor)
        orbitResponse_x = []
        orbitResponse_y = []
        madx.globals.update(self.data['model'])
        # We update the model given the parameter list
        # before we compute the Orbit Response
        for p in range(len(pList)):
            madx.globals[pList[p]] += dpList[p]

        twiss1 = madx.twiss(sequence=self.sequence, RMatrix=True,
                            alfx = -7.2739282, alfy = -0.23560719,
                            betx = 44.404075,  bety = 3.8548009,
                            x = self.dx, y = self.dy,
                            px = self.dpx, py = self.dpy)
        x1 = twiss1.x[iMonitor]
        y1 = twiss1.y[iMonitor]
        for k in self.kickers:
            madx.globals.update(self.data['model'])
            # We update the model given the parameter list
            # before we compute the Orbit Response
            for p in range(len(pList)):
                madx.globals[pList[p]] += dpList[p]
                madx.globals[k] += kick
                
            twiss2 = madx.twiss(sequence=self.sequence, RMatrix=True,
                                alfx = -7.2739282, alfy = -0.23560719,
                                betx = 44.404075,  bety = 3.8548009,
                                x = self.dx, y = self.dy,
                                px = self.dpx, py = self.dpy)
            x2 = twiss2.x[iMonitor]
            y2 = twiss2.y[iMonitor]
            # Horizontal and vertical response
            cx = (x2 - x1) / kick
            cy = (y2 - y1) / kick
            orbitResponse_x.append(cx)
            orbitResponse_y.append(cy)
        return np.array(orbitResponse_x), np.array(orbitResponse_x)

    def forwardDorm(self, pList, dpList, kick=2e-4, dp=1e-9):
        """
        Computes the orbit response derivative for a given
        list of parameters with a forward finite difference
        @param pList is the list of parameters the derivative should
               be computed to
        @param dp0List are the absolute changes around the measurement 
               nominal value for the parameter
        @param dp0 cannot be smaller than 1e-9 for numerical precision
               reasons!!! Going smaller changes drastically the computation.
               Going up leaves the same results up to numerical precision.
        """
        dpList = np.array(dpList)
        ormMx1, ormMy1 = self.ormModelpList(pList, dpList, kick)
        dCx = []
        dCy = []
        for param in range(len(pList)):
            # Working copy of the parameter list
            dpList_cp = np.array(dpList)
            dpList_cp[param] += dp
            ormMx2, ormMy2 = self.ormModelpList(pList, dpList_cp, kick)
            dCxdP = (ormMx2 - ormMx1) / dp
            dCydP = (ormMy2 - ormMy1) / dp
            dCx.append(dCxdP)
            dCy.append(dCydP)
        return np.array(dCx), np.array(dCy)

    def backwardDorm(self, pList, dpList, kick=2e-4, dp=1e-9):
        """
        Backward finite difference (See forwardDorm for description)
        """
        dpList = np.array(dpList)
        ormMx1, ormMy1 = self.ormModelpList(pList, dpList, kick)
        dCx = []
        dCy = []
        for param in range(len(pList)):
            # Working copy of the parameter list
            dpList_cp = np.array(dpList)
            dpList_cp[param] -= dp
            ormMx2, ormMy2 = self.ormModelpList(pList, dpList_cp, kick)
            dCxdP = (ormMx1 - ormMx2) / dp
            dCydP = (ormMy1 - ormMy2) / dp
            dCx.append(dCxdP)
            dCy.append(dCydP)
        return np.array(dCx), np.array(dCy)

    def centralDorm(self, pList, dpList, kick=2e-4, dp=1e-5):
        """
        Central finite difference. (See forwardDorm for description).
        This is the way to go.
        """
        dpList = np.array(dpList)
        dCx = []
        dCy = []
        for param in range(len(pList)):
            # Working copy of the parameter list
            dpList_back = np.array(dpList)
            dpList_forw = np.array(dpList)
            dpList_back[param] -= dp
            dpList_forw[param] += dp
            ormMxback, ormMyback = self.ormModelpList(pList, dpList_back, kick)
            ormMxforw, ormMyforw = self.ormModelpList(pList, dpList_forw, kick)
            dCxdP = (ormMxforw - ormMxback) / (2*dp)
            dCydP = (ormMyforw - ormMyback) / (2*dp)
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
        ormG, dormG   = self.ormMeasured()
        ormMx, ormMy  = self.ormModel()
        # Horizontal response
        y1  = ormG[0]
        dy1 = dormG[0]
        # Vertical response
        y2  = ormG[1]
        dy2 = dormG[1]
        if messFile2 != '':
            data2           = self.readData(messFile2)
            ormG2, dormG2   = self.ormMeasured(data2)
            y12  = ormG2[0]
            dy12 = dormG2[0]
            y22  = ormG2[1]
            dy22 = dormG2[1]
        self._plotData(y1, dy1, y2, dy2, ormMx, ormMy)
        if(messFile2!=''):
            self._plotData(y12, dy12, y22, dy22,
                          ormMx, ormMy,plotModel=False)
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def _plotData(self, y1, dy1, y2, dy2, ormMx, ormMy,
                 save = False, plotModel = True):
        """
        Just for inner functionality
        """
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

class ORMOptimizer:

    def __init__(self, messFiles, madxFile, readOrm=False, plotEachM=True):
        self.messFiles = messFiles
        self.madxFile  = madxFile
        self.orbitResponseAnalyzer = OrbitResponse(messFiles[0], madxFile)
        self.monitors  = self.getMonitors()
        self.kickers   = {}
        self.nMonitors = len(self.monitors)
        self.nParams   = 0
        # We can spare some time if the ORM has already been computed
        # ormMx stands for Orbit Response Matrix Measurement in x-plane
        if(readOrm):
            self.ormMx, self.ormMy = self.readMessOrm()
            self.setKickers()
        else:
            self.ormMx, self.ormMy = self.initializeOrm(True, plotEachM)
        self.nKickers = len(self.kickers)
        self.dCij_x = 0
        self.dCij_y = 0
        self.Ax = 0
        self.Ay = 0
        # Needed for optimize2
        self.dp0 = np.zeros(self.nMonitors + self.nKickers)
        self.dOrmxdP = []
        self.dOrmydP = []
        self.pList = []
        self.dp0List = []
        self.y0 = []
        
    def getMonitors(self):
        orbResponse = self.orbitResponseAnalyzer
        monitors = []
        for measurement in self.messFiles:
            orbResponse.setData(measurement)
            monitors.append(orbResponse.getMonitor())
        return monitors
            
    def initializeOrm(self, write=True, showPlots=False):
        """
        Computes the measured orbit response and the corresponding
        modeled response
        @param (bool)write if True, writes the measured orbit response
               together with the initial modeld response
        @param (bool)showPlots if True, displays the measured and modeled
               orbit response for each measurement
        """
        orbitResponse_x = []
        orbitResponse_y = []
        orA = self.orbitResponseAnalyzer
        for mon_i in range(len(self.messFiles)):
            measurement = self.messFiles[mon_i]
            orA.setData(measurement)
            kickers =  orA.kickers
            for k in kickers:
                if k not in self.kickers: self.kickers[k] = len(self.kickers) + 1
            mMeasured, dmMeasured = orA.ormMeasured()
            mModelx,   mModely    = orA.ormModel()
            mMx  = mMeasured[0]
            mMy  = mMeasured[1]
            dmMx = dmMeasured[0]
            dmMy = dmMeasured[1]
            if(showPlots): orA.visualizeData()
            for k_i in range(len(kickers)):
                orbitResponse_x.append([mon_i, k_i, mMx[k_i], dmMx[k_i], mModelx[k_i]])
                orbitResponse_y.append([mon_i, k_i, mMy[k_i], dmMy[k_i], mModely[k_i]])
        if(write):
            head = 'Monitor  Kicker   orm   dorm   madxOrm'
            np.savetxt('ormx.txt',orbitResponse_x, fmt=['%i','%i','%e', '%e','%e'],
                       delimiter='  ', header=head)
            np.savetxt('ormy.txt',orbitResponse_y, fmt=['%i','%i','%e', '%e','%e'],
                       delimiter='  ', header=head)
        return orbitResponse_x, orbitResponse_y

    def setdCij(self):
        """
        Computes the difference between measured and modeled OR
        Requires self.ormMx and self.ormMy
        """
        dCij_x = self.ormMx
        dCij_y = self.ormMy
        dCij_x = np.transpose(dCij_x)
        dCij_y = np.transpose(dCij_y)
        dCij_x = (dCij_x[2]-dCij_x[4])/dCij_x[3]
        dCij_y = (dCij_y[2]-dCij_y[4])/dCij_y[3]
        self.dCij_x = dCij_x
        self.dCij_y = dCij_y
        
    def setKickers(self):
        """
        This is for the case where we just read the data. 
        We build a dictionary of kickers from the last monitor
        """
        ormA =  self.orbitResponseAnalyzer
        ormA.setData(self.messFiles[-1])
        ks   =  ormA.kickers
        for k in ks:
            if k not in self.kickers: self.kickers[k] = len(self.kickers) + 1

    def readMessOrm(self):
        """
        Read the ORM already computed
        """
        ormx = np.loadtxt('ormx.txt')
        ormy = np.loadtxt('ormy.txt')
        return ormx, ormy

    def initializeAMat(self, visualize=False):
        """
        Initializes the Matrix A (s. Appendix A in protocol)
        """
        # ormMx has the structure (see initializeOrm() )
        # Monitor Kicker ORMeasured Error ORModeled
        nColumns = len(self.ormMx)
        nRows    = self.nMonitors + self.nKickers
        Ax = np.zeros((nColumns, nRows))
        Ay = np.zeros((nColumns, nRows))
        for i in range(nColumns):
            Cij_x = self.ormMx[i]
            Cij_y = self.ormMy[i]
            # Normalization
            entryCij_x = Cij_x[4] / Cij_x[3]
            entryCij_y = Cij_y[4] / Cij_y[3]
            mon_i  = int(Cij_x[0])
            kick_j = int(Cij_x[1])
            Ax[i][mon_i] = -entryCij_x
            Ay[i][mon_i] = -entryCij_y
            Ax[i][self.nMonitors + kick_j] = entryCij_x
            Ay[i][self.nMonitors + kick_j] = entryCij_y

        self.setdCij()

        if(len(self.dOrmxdP)):
            dOrmx = np.transpose(self.dOrmxdP)
            dOrmy = np.transpose(self.dOrmydP)
            for i in range(nColumns):
                dOrmx[i] = dOrmx[i]/Cijx[3]
                dOrmy[i] = dOrmy[i]/Cijy[3]
                """
                for e in range(len(dOrmx[i])):
                    if dOrmx[i][e] > 10: dOrmx[i][e] = 0
                    if dOrmy[i][e] > 10: dOrmy[i][e] = 0
                """
            Ax = np.hstack((dOrmx,Mx))
            Ay = np.hstack((dOrmy,My))
        # In case we want to see that everything works fine.
        # Difficult to see though, since we have an ~50x(7+19)
        # matrix.
        if(visualize):
            print('nMon x nKickers = {}'.format(self.nMonitors*self.nKickers))
            print('nMon + nKickers = {}'.format(self.nMonitors+self.nKickers))
            print('Horizontal:')
            for i in Ax:
                for j in i:
                    print(round(j,4), end='  ')
                print()
            print('Vertical')
            for i in Ay:
                for j in i:
                    print(round(j,4), end='  ')
                print()
        self.Ax = Ax
        self.Ay = Ay

    def fitParameters(self, singularMask=100):
        """
        Here the parameters are fitted as in
        XFEL Beam Dynamics meeting (12.12.2005)
        @param singularMask controls the entries that will be taken into
               account in the singular value decomposition.
        """
        self.setdCij()
        # dp = (B+)*A^{T}*(ORM_{gemessen} - ORM_{model})
        # B  = A^{T}*A
        # B+ = V*D*U^{T} from SVD
        At_x = np.transpose(self.Ax)
        At_y = np.transpose(self.Ay)
        Bx   = np.matmul(At_x, self.Ax)
        By   = np.matmul(At_y, self.Ay)    
        Bp_x = self._getPseudoInverse(Bx, singularMask)
        Bp_y = self._getPseudoInverse(By, singularMask)
        err_x = np.matmul(Bp_x, np.matmul(At_x, self.dCij_x))
        err_y = np.matmul(Bp_y, np.matmul(At_y, self.dCij_y))
        return err_x, err_y
        
    def _getPseudoInverse(self, B, singularMask):
        """
        Internal computation. Just for style...
        """
        U, S, Vt = np.linalg.svd(B, full_matrices=False, compute_uv=True)
        for s in range(len(S)):
            if S[s] < singularMask: S[s] = 10e15
        Ut = np.transpose(U)
        V  = np.transpose(Vt)
        D  = 1/S
        D  = np.diag(D)
        return np.matmul(V, np.matmul(D, Ut))

    def actualizeModelSimple(self, errx, erry):
        """
        This function actualizes the model in self.ormMx and self.ormMy
        We scale the parameters, monitor reads and the kickers
        according to Eqs. 8 and 9
        """
        nMon   = self.nMonitors
        mKicks = self.nKickers
        kicksErr = ( errx[nMon:] + erry[nMon:] ) / 2
        for j in range(mKicks):
            errx[nMon+j] = kicksErr[j]
            erry[nMon+j] = kicksErr[j]
        for c in self.ormMx:
            f = ( ( 1 + errx[self.nMonitors + int(c[1])] ) /
                  ( 1 + errx[int(c[0])] ) )
            c[4] *= f

        for c in self.ormMy:
            f = ( ( 1 + erry[self.nMonitors + int(c[1])] ) /
                  ( 1 + erry[int(c[0])] ) )
            c[4] *= f
        self.setdCij()
        return errx[:nMon],erry[:nMon], kicksErr

    def getChi2(self):
        """
        Returns the Chi2 sum
        """
        chi2x = sum(self.dCij_x)**2
        chi2y = sum(self.dCij_y)**2
        return chi2x, chi2y
    
    def fitErrorsSimple(self, error=1e-9 ,plot=True, maxIt=100):
        """
        Fits the errors until a user defined precision 
        is reached
        """
        self.initializeAMat()
        chi2 = [self.getChi2()]
        fitErrx, fitErry = self.getChi2()
        monx, mony, kickerErr = [], [], []
        it = 0
        if(plot):
            ormMx   = np.transpose(self.ormMx)
            ormMy   = np.transpose(self.ormMy)
            modely0 = ormMy[4]
            modelx0 = ormMx[4]
        while(fitErrx > error and maxIt > it):
            chix_old, chiy_old = self.getChi2()
            errx, erry = self.fitParameters()
            x,y,z = self.actualizeModelSimple(errx, erry)
            kickerErr.append(z)
            monx.append(x)
            mony.append(y)
            chi2.append(self.getChi2())
            chix_new, chiy_new = self.getChi2()
            fitErrx = abs(chix_old - chix_new)
            it += 1
        if(plot):
            chi2 = np.transpose(chi2)
            plt.plot(abs(chi2[0][1:]-chi2[0][:-1]),marker='.',label=r'$\chi^2_x$')
            plt.plot(abs(chi2[1][1:]-chi2[1][:-1]),marker='.',label=r'$\chi^2_y$')
            plt.xlabel('Iteration')
            plt.ylabel(r'Error ($|\chi^2_{new} - \chi^2_{old}$|)')
            plt.yscale('log')
            plt.legend()
            plt.show()
            self.plotSimpleFit(modelx0,modely0)
        kickerErr = np.transpose(kickerErr)
        relFit = np.ones(self.nKickers)
        for k in range(self.nKickers):
            for c in kickerErr[k]:
                relFit[k] *= (1 + c)
        labls = list(self.kickers.keys())
        print('Fit converged')
        print('')
        print('Number of iterations: ', it)
        print('')
        print('Initial chi2: x-> ', chi2[0][0], ' y-> ', chi2[1][0])
        print('')
        print('Final chi2: x-> ', chi2[0][-1],  ' y-> ', chi2[1][-1])
        print('')
        print('Fit parameters for kickers:')
        print('')
        for k in range(self.nKickers):
            fit_k = (relFit[k]-1)*100
            print('{} : {} %'.format(labls[k], round(fit_k,3)))
        print('')
        print('Fit parameters for monitors')
        print('')
        monxErr = np.transpose(monx)
        monyErr = np.transpose(mony)
        relFitx = np.ones(self.nMonitors)
        relFity = np.ones(self.nMonitors)
        for m in range(self.nMonitors):
            for c in monxErr[m]:
                relFitx[m] *= (1 + c)
            for c in monyErr[m]:
                relFity[m] *= (1 + c)
        for m in range(self.nMonitors):
            fitx_m = (relFitx[m]-1)*100
            fity_m = (relFity[m]-1)*100
            print('{} : x-> {} %  y-> {} %'.format(self.monitors[m],
                                                 round(fitx_m,3),
                                                 round(fity_m,3)))
        
    def plotSimpleFit(self, modelx0, modely0):
        ormMx = np.transpose(self.ormMx)
        measured = ormMx[2]
        error    = ormMx[3]
        model    = ormMx[4]
        x        = np.arange(len(measured))
        plt.errorbar(x, measured, yerr=error, label='Messung')
        plt.plot(x, modelx0, label='Model',marker='.')
        plt.plot(x, model, label='Fit', marker='.')
        plt.legend(loc=0)
        plt.show()
        ormMy = np.transpose(self.ormMy)
        measured = ormMy[2]
        error    = ormMy[3]
        model    = ormMy[4]
        y        = np.arange(len(measured))
        plt.errorbar(y, measured, yerr=error, label='Messung')
        plt.plot(y, modely0, label='Model', marker='.')
        plt.plot(y, model, label='Fit', marker='.')
        plt.legend(loc=0)
        plt.show()
        
    def computedOrmdP(self):
        """
        Computes the orbit response matrix derivative
        """
        dOrmx = []
        dOrmy = []

        #Initialize dp0 with zeros
        if len(self.dp0List) == 0: self.dp0List = np.zeros(len(self.pList))

        for measurement in self.messFiles:
            ormA = ormAna.OrmAnalysis( measurement, self.madxFile )
            dormx, dormy = ormA.computeDorm(self.pList, self.dp0List)
            #print(dormx)
            dOrmx.append(dormx)
            dOrmy.append(dormy)

        dormx = []
        dormy = []
        for j in range( self.nParams ):
            myArrax = []
            myArray = []
            for i in range( len(self.messFiles) ):
                for k in dOrmx[i][j]:
                    myArrax.append(k)
                for k in dOrmy[i][j]:
                    myArray.append(k)
            dormx.append(myArrax)
            dormy.append(myArray)
            
        self.dOrmxdP = dormx
        self.dOrmydP = dormy
        
    def optimize1(self, plot=True, singularMask=1e-4):
        """
        Computes the optimization scheme with help of the
        function fitParameters and is able to plot the results.
        It prints out the computed fit parameters for the corrector
        kickers and the monitors.
        @param plot is a boolean, if True, the horizontal and vertical
               orbit response will be plotted together with the initial
               MAD-X model and the fitted orbit response.
        @param singularMask see function fitParameters()
        """
        L2 = []
        kickers = list(self.kickers.keys())
        xlabels = []
        monPos = []

        Mx = np.transpose(self.ormMx)
        My = np.transpose(self.ormMy)

        if(plot):
            for i in range (len(Mx[1])):
                xlabels.append(kickers[int(Mx[1][i])])
                if (Mx[1][i] == 0): monPos.append(i)

            x = np.linspace(0,len(Mx[2]),len(Mx[2]))
            plt.figure(1)
            plt.errorbar(x,Mx[2],yerr=Mx[3],marker='.',label='xMeasured')
            plt.plot(x,Mx[4],label='xModel',marker='.')
            locs, labels = plt.xticks()
            plt.xticks(x, xlabels, rotation='vertical')
            plt.legend(loc=0)
            for i in monPos:
                plt.axvline(i,linestyle='--')

            plt.figure(2)
            plt.errorbar(x,My[2],yerr=Mx[3],marker='.',label='yMeasured')
            plt.plot(x,My[4],label='yModel',marker='.')
            plt.legend(loc=0)
            for i in monPos:
                plt.axvline(i,linestyle='--')

        kickerFits = []
        monFits = []
        kickNames = list(self.kickers.keys())
        monNames  = list(self.monitors.keys())

        monx = self.dp0[:self.nMonitors]
        kickx = self.dp0[self.nMonitors:]
        kickerFits.append(kickx)
        monFits.append(monx)
        chi20 = self.fitParameters(singularMask)
        L2.append(chi20)
        error = 1

        while(error > 1e-10):
            monx = self.dp0[:self.nMonitors]
            kickx = self.dp0[self.nMonitors:]
            kickerFits.append(kickx)
            monFits.append(monx)
            chi21 = self.fitParameters(singularMask)
            error = abs(chi21 - chi20)
            chi20 = chi21
            L2.append(chi21)

        kickerFits = np.transpose(kickerFits)
        kFits = []
        print("Kickers:")
        for i in range(len(kickerFits)):
            print(kickNames[i],' : ',round(sum(kickerFits[i]),4))
            kFits.append(sum(kickerFits[i]))

        monFits = np.transpose(monFits)
        mFits = []
        print("Monitors:")
        for i in range(len(monFits)):
            print(monNames[i]," : " ,round(sum(monFits[i]),4))
            mFits.append(sum(monFits[i]))

        Mx = np.transpose(self.ormMx)
        My = np.transpose(self.ormMy)

        if(plot):
            plt.figure(1)
            plt.plot(x,Mx[4],label='xFit',marker='.')
            plt.ylabel(r'Horizontal orbit response [mm mrad$^{-1}$]')
            plt.legend(loc=0)
            plt.figure(2)
            plt.plot(x,My[4],label='yFit',marker='.')
            plt.legend(loc=0)
            plt.show()

            plt.plot(L2,marker='.')
            plt.xlabel('Iteration number')
            plt.ylabel(r'$||\vec{y}_1 - \vec{y}_0||_2$')
            plt.yscale('log')
            plt.show()

        return np.array(mFits), np.array(kFits)

    def optimize2(self, pList, plot=True, singularMask=1e-4, nIterations=1):
        """
        Computes the optimization scheme with help of the
        function fitParameters and is able to plot the results.
        It prints out the computed fit parameters for the corrector
        kickers and the monitors.
        It also takes into consideration a list of given lattice parameters
        in the fit.
        @param plot is a boolean, if True, the horizontal and vertical
               orbit response will be plotted together with the initial
               MAD-X model and the fitted orbit response.
        @param singularMask see function fitParameters()
        """
        self.pList = pList
        self.nParams = len(pList)
        self.computedOrmdP()
        self.computeOrm()
        L2 = []

        kickers = list(self.kickers.keys())
        xlabels = []
        monPos = []

        Mx = np.transpose(self.ormMx)
        My = np.transpose(self.ormMy)

        if(plot):
            for i in range (len(Mx[1])):
                xlabels.append(kickers[int(Mx[1][i])])
                if (Mx[1][i] == 0): monPos.append(i)

            x = np.linspace(0,len(Mx[2]),len(Mx[2]))
            plt.figure(1)
            plt.errorbar(x,Mx[2],yerr=Mx[3],marker='.',label='xMeasured')
            plt.plot(x,Mx[4],label='xModel',marker='.')
            locs, labels = plt.xticks()
            plt.xticks(x, xlabels, rotation='vertical')
            plt.legend(loc=0)
            for i in monPos:
                plt.axvline(i,linestyle='--')

            plt.figure(2)
            plt.errorbar(x,My[2],yerr=Mx[3],marker='.',label='yMeasured')
            plt.plot(x,My[4],label='yModel',marker='.')
            plt.legend(loc=0)
            for i in monPos:
                plt.axvline(i,linestyle='--')

        kickerFits = []
        monFits = []
        kickNames = list(self.kickers.keys())
        monNames  = list(self.monitors.keys())

        print('Number of fit parameters: ', self.nParams)
        print('Number of monitors: ', len(monNames))
        print('Number of kickers: ', len(kickNames))
        print('Fitting model to data. This might take a while...')
        error = 10000
        iteration = 0
        for iteration in range(nIterations):
        #while( error > 1e-1 ):
            print('Iteration ',iteration+1,' , ERROR = ', error)
            error = self.fitParameters(singularMask)
            L2.append(error)
            monx = self.dp0[self.nParams:self.nParams+len(monNames)]
            kickx = self.dp0[self.nParams+len(monNames):]
            kickerFits.append(kickx)
            monFits.append(monx)
            self.computedOrmdP()
            self.computeOrm()
            #iteration += 1

        kickerFits = np.transpose(kickerFits)
        kFits = []
        print("Kickers:")
        for i in range(len(kickerFits)):
            print(kickNames[i],' : ',round(sum(kickerFits[i]),4))
            kFits.append(sum(kickerFits[i]))
        print()

        monFits = np.transpose(monFits)
        mFits = []
        print("Monitors:")
        for i in range(len(monFits)):
            print(monNames[i]," : " ,round(sum(monFits[i]),4))
            mFits.append(sum(monFits[i]))
        print()

        print('Parameters:')
        for i in range(len(self.pList)):
            print(self.pList[i], ' : ', round(self.dp0[i],5))
        print()

        Mx = np.transpose(self.ormMx)
        My = np.transpose(self.ormMy)

        if(plot):
            plt.figure(1)
            plt.plot(x,Mx[4],label='xFit',marker='.')
            plt.ylabel(r'Horizontal orbit response [mm mrad$^{-1}$]')
            plt.legend(loc=0)
            plt.figure(2)
            plt.plot(x,My[4],label='yFit',marker='.')
            plt.legend(loc=0)
            plt.show()

            plt.plot(L2,marker='.')
            plt.xlabel('Iteration number')
            plt.ylabel(r'$||\vec{y}_1 - \vec{y}_0||_2$')
            plt.yscale('log')
            plt.show()

        return np.array(mFits), np.array(kFits)
