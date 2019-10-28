from cpymad.madx import Madx
from yaml import safe_load
from ProfileAnalyzer import ProfileAnalyzer

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
    def __init__(self, dataFile, madxModelFile, profilePath):
        self.madxModelFile       = madxModelFile
        self.dataFile            = dataFile
        self.profilePath         = profilePath
        self.data                = self.readData(dataFile)
        self.monitor             = self.getMonitor()
        self.kickers, self.kicks = self.getKicks()
        self.sequence            = self.getSequence()
        self.madx                = Madx(stdout=False)
        self.madx.call(file=self.madxModelFile, chdir=True)

        # This are the initial conditions for the Twiss Module of MAD-X
        # there doesn't seem to be a strong dependence on them
        self.dx  = 1.0e-4
        self.dpx = 1.0e-6
        self.dy  = 1.0e-4
        self.dpy = 1.0e-6

        # Initial twiss parameters
        self.alfax = -7.2739282
        self.alfay = -0.23560719
        self.betax = 44.404075
        self.betay = 3.8548009 

    def readData(self, dataFile):
        """
        Reads data from the yaml file as written by madgui
        and returns an ordered dictionary
        """
        with open(dataFile) as f: data = safe_load(f)
        knobs = data['knobs']
        # Normal ordering for kickers.
        # Namely, according to their position s
        # and in increasing order
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
    
        profAnalizer = ProfileAnalyzer(self.data, self.profilePath)
        profAnalizer.fitProfiles(self.monitor.upper(), showProfiles=False,
                                 skipShots=1, plot=False)
        pOrmx, pOrmy = profAnalizer.messDatax, profAnalizer.messDatay
        gridProfiles = (len(pOrmx) != 0)
        print(' GridProfiles: ', gridProfiles)
        
        records     = self.data['records']
        beamMess    = []
        beamMessErr = []
        for messung in records:
            shots    = messung['shots']
            dataBeam = []
            for shot in shots: dataBeam.append(shot[self.monitor])
            mean   = np.mean(dataBeam, axis = 0)
            stdDev = np.std (dataBeam, axis = 0)
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
            orbR_k = (beamMess[k+1] - beamMess[0]) / kickDiff
            orbRErr_k = np.sqrt(beamMessErr[k+1]**2 + beamMessErr[0]**2) \
                        / kickDiff
            if(gridProfiles):
                # Grid horizontal axis is inverted
                orbR_kx = -(pOrmx[self.kickers[k]][0] - pOrmx[''][0]) / kickDiff
                orbR_ky = (pOrmy[self.kickers[k]][0] - pOrmy[''][0]) / kickDiff
                # Gaussian error propagation
                orbRErr_kx = np.sqrt(pOrmx[self.kickers[k]][1]**2 + pOrmx[''][1]**2) \
                             / kickDiff
                orbRErr_ky = np.sqrt(pOrmy[self.kickers[k]][1]**2 + pOrmy[''][1]**2) \
                             / kickDiff
                orbitResponse.append([orbR_kx, orbR_ky])
                orbitResponseErr.append([orbRErr_kx, orbRErr_ky])
            else:
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
                            alfx=self.alfax, alfy=self.alfay,
                            betx=self.betax, bety=self.betay,
                            x=self.dx, y=self.dy,
                            px=self.dpx, py=self.dpy)
        x0 = twiss0.x[iMonitor]
        y0 = twiss0.y[iMonitor]
        orbitResponse_x = []
        orbitResponse_y = []
        for k in self.kickers:
            madx.globals.update(self.data['model'])
            madx.globals[k] += kick
            twiss1 = madx.twiss(sequence=self.sequence, RMatrix=True,
                                alfx=self.alfax, alfy=self.alfay,
                                betx=self.betax, bety=self.betay,
                                x=self.dx, y=self.dy,
                                px=self.dpx, py=self.dpy)
            x1 = twiss1.x[iMonitor]
            y1 = twiss1.y[iMonitor]
            # Horizontal and vertical response
            cx = (x1 - x0) / kick
            cy = (y1 - y0) / kick
            orbitResponse_x.append(cx)
            orbitResponse_y.append(cy)
        return orbitResponse_x, orbitResponse_y

    def ormModelpList(self, pList, dpList,
                      kickers, dkickers,
                      dmonx, dmony,
                      kick=2e-4):
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
        for p in range(len(pList)):   madx.globals[pList[p]] += dpList[p]
        for k in range(len(kickers)): madx.globals[kickers[k]] /= dkickers[k]
        twiss1 = madx.twiss(sequence=self.sequence, RMatrix=True,
                            alfx=self.alfax, alfy=self.alfay,
                            betx=self.betax, bety=self.betay,
                            x=self.dx, y=self.dy,
                            px=self.dpx, py=self.dpy)
        x1 = twiss1.x[iMonitor]
        y1 = twiss1.y[iMonitor]
        for k_i in range(len(self.kickers)):
            k = self.kickers[k_i]
            madx.globals.update(self.data['model'])
            # We update the model again. (To reset the Twiss command)
            for p in range(len(pList)): madx.globals[pList[p]] += dpList[p]
            for kname in range(len(kickers)): madx.globals[kickers[kname]] /= (dkickers[kname])
            if len(dkickers): madx.globals[k] = madx.globals[k] + kick/dkickers[k_i]
            else: madx.globals[k] += kick
            twiss2 = madx.twiss(sequence=self.sequence, RMatrix=True,
                                alfx=self.alfax, alfy=self.alfay,
                                betx=self.betax, bety=self.betay,
                                x=self.dx, y=self.dy,
                                px=self.dpx, py=self.dpy)
            x2 = twiss2.x[iMonitor]
            y2 = twiss2.y[iMonitor]
            # Horizontal and vertical response
            cx = (x2 - x1) / (kick * dmonx)
            cy = (y2 - y1) / (kick * dmony)
            orbitResponse_x.append(cx)
            orbitResponse_y.append(cy)
        return np.array(orbitResponse_x), np.array(orbitResponse_y)

    def centralDorm(self, pList, dpList, kickers=[], dkickers=[],
                    dmonx=1, dmony=1, kick=2e-4, dp=1e-5):
        """
        Central finite difference
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
            ormMxback, ormMyback = self.ormModelpList(pList, dpList_back,
                                                      kickers, dkickers,
                                                      dmonx, dmony,
                                                      kick)
            ormMxforw, ormMyforw = self.ormModelpList(pList, dpList_forw,
                                                      kickers, dkickers,
                                                      dmonx, dmony,
                                                      kick)
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
                          ormMx, ormMy, plotModel=False)
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def _plotData(self, y1, dy1, y2, dy2, ormMx, ormMy,
                 save=False, plotModel=True):
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

    def __init__(self, messFiles, madxFile, profilePath,
                 readOrm=False, plotEachM=True, savePath='.'):
        self.messFiles = messFiles
        self.madxFile  = madxFile
        self.orbitResponseAnalyzer = OrbitResponse(messFiles[0], madxFile, profilePath)
        self.monitors  = self.getMonitors()
        self.kickers   = {}
        self.nMonitors = len(self.monitors)
        self.nParams   = 0
        self.savePath  = savePath
        self.ormMx, self.ormMy = self.initializeOrm(False)
        self.nKickers = len(self.kickers)
        self.dCij_x = 0
        self.dCij_y = 0
        self.setdCij()
        self.Ax = 0
        self.Ay = 0
        # Needed for the fit
        self.singularMask = 100
        self.dp0 = np.zeros(self.nMonitors + self.nKickers)
        self.dOrmxdP = []
        self.dOrmydP = []
        self.pList   = []
        self.dp0List = []
        self.dParams = []
        self.dKickers = np.ones(len(self.kickers))
        self.dMonitors_x = np.ones(self.nMonitors)
        self.dMonitors_y = np.ones(self.nMonitors)

    def getMonitors(self):
        orbResponse = self.orbitResponseAnalyzer
        monitors = []
        for measurement in self.messFiles:
            orbResponse.setData(measurement)
            monitors.append(orbResponse.getMonitor())
        return monitors

    def initializeOrm(self, showPlots=False):
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
        return orbitResponse_x, orbitResponse_y

    def setdCij(self):
        """
        Computes the difference between measured and modeled Orbit Response
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

    def initializeAMat(self, visualize=False):
        """
        Initializes the Matrix A (s. Appendix A in protocol)
        """
        # ormMx has the structure (see initializeOrm())
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

        if(len(self.dOrmxdP)):
            dOrmx = np.transpose(self.dOrmxdP)
            dOrmy = np.transpose(self.dOrmydP)
            for i in range(nColumns):
                dOrmx[i] = dOrmx[i]/Cij_x[3]
                dOrmy[i] = dOrmy[i]/Cij_y[3]
            Ax = np.hstack((dOrmx,Ax))
            Ay = np.hstack((dOrmy,Ay))
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

    def fitParameters(self):
        """
        Here the parameters are fitted as in
        XFEL Beam Dynamics meeting (12.12.2005)
        """
        # dp = (B+)*A^{T}*(ORM_{gemessen} - ORM_{model})
        # B  = A^{T}*A
        # B+ = V*D*U^{T} from SVD
        At_x = np.transpose(self.Ax)
        At_y = np.transpose(self.Ay)
        Bx   = np.matmul(At_x, self.Ax)
        By   = np.matmul(At_y, self.Ay)
        Bp_x = self._getPseudoInverse(Bx)
        Bp_y = self._getPseudoInverse(By)
        err_x = np.matmul(Bp_x, np.matmul(At_x, self.dCij_x))
        err_y = np.matmul(Bp_y, np.matmul(At_y, self.dCij_y))
        return err_x, err_y

    def _getPseudoInverse(self, B):
        """
        Internal computation.
        singularMask controls the entries that will be taken into
        account in the singular value decomposition.
        """
        U, S, Vt = np.linalg.svd(B, full_matrices=False, compute_uv=True)
        for s in range(len(S)):
            if S[s] < self.singularMask: S[s] = 10e30
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

    def getGlobalMachineError(self):
        gmex = sum(self.dCij_x**2)
        gmey = sum(self.dCij_y**2)
        return gmex, gmey

    def fitErrorsSimple(self, error=1e-9, plot=True, maxIt=100, singularMask=0):
        """
        Fits the errors until a user defined precision
        is reached. Takes into consideration the Global Machine
        Error (gme)
        """
        if(singularMask): self.singularMask = singularMask
        self.initializeAMat()
        gme = [self.getGlobalMachineError()]
        fitErrx, fitErry = self.getGlobalMachineError()
        monxErrFit, monyErrFit, kickerErrFit = [], [], []
        it = 0
        if(plot):
            ormMx   = np.transpose(self.ormMx)
            ormMy   = np.transpose(self.ormMy)
            modely0 = ormMy[4]
            modelx0 = ormMx[4]
        converged = False
        while(not converged and maxIt > it):
            gmex_old, gmey_old = self.getGlobalMachineError()
            errx, erry = self.fitParameters()
            x,y,z = self.actualizeModelSimple(errx, erry)
            kickerErrFit.append(z)
            monxErrFit.append(x)
            monyErrFit.append(y)
            gme.append(self.getGlobalMachineError())
            gmex_new, gmey_new = self.getGlobalMachineError()
            fitErrx   = abs(gmex_old - gmex_new)
            converged = (fitErrx < error)
            it += 1
        self._displayResults(kickerErrFit,monxErrFit,monyErrFit,
                             it, gme, converged)
        self._plotFit(modelx0,modely0)

    def actualizeModel(self, errx, erry):
        paramErrx = errx[:self.nParams]
        monErrx   = errx[self.nParams:self.nParams+len(self.messFiles)]
        kickErrx  = errx[self.nParams+len(self.messFiles):]
        paramErry = erry[:self.nParams]
        monErry   = erry[self.nParams:self.nParams+len(self.messFiles)]
        kickErry  = erry[self.nParams+len(self.messFiles):]

        # This can still be optimized according to necesity
        paramErr = (paramErrx + paramErry) / 2
        kicksErr = (kickErrx + kickErry) / 2

        for i in range(len(self.kickers)): self.dKickers[i] *= (1 + kicksErr[i])
        for i in range(self.nParams): self.dp0List[i] += paramErr[i]
        for i in range(self.nMonitors):
            self.dMonitors_x[i] *= (1 + monErrx[i])
            self.dMonitors_y[i] *= (1 + monErry[i])
        orA = self.orbitResponseAnalyzer
        orbitResponse_x = []
        orbitResponse_y = []
        for mon_i in range(len(self.messFiles)):
            measurement = self.messFiles[mon_i]
            dmonx = self.dMonitors_x[mon_i]
            dmony = self.dMonitors_y[mon_i]
            orA.setData(measurement)
            kickers = orA.kickers
            for k in kickers:
                if k not in self.kickers: self.kickers[k]=len(self.kickers) + 1
            mModelx, mModely = orA.ormModelpList(self.pList, self.dp0List,
                                                 list(self.kickers.keys()),
                                                      self.dKickers,
                                                 dmonx, dmony)
            for k_i in range(len(kickers)):
                orbitResponse_x.append(mModelx[k_i])
                orbitResponse_y.append(mModely[k_i])

        for c in range(len(self.ormMx)):
            self.ormMx[c][4] = orbitResponse_x[c]
            self.ormMy[c][4] = orbitResponse_y[c]
        self.setdCij()

    def fitErrors(self, pList, singularMask=0,
                  maxIt=30, error=1e-3,
                  continueFit=False):
        if(singularMask): self.singularMask = singularMask
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        print('        Initializing fitting')
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        ormMx   = np.transpose(self.ormMx)
        ormMy   = np.transpose(self.ormMy)
        modely0 = ormMy[4]
        modelx0 = ormMx[4]
        
        gme = [self.getGlobalMachineError()]
        self.setpList(pList)
        self.computedOrmdP()
        self.initializeAMat()
        converged = False
        it = 0

        while (not converged and it < maxIt):
            gmex_old, gmey_old = self.getGlobalMachineError()
            print('    Iteration:   ', it+1)
            gmex = round(gmex_old,3)
            gmey = round(gmey_old,3)
            gmexy  = round(gmex + gmey,3)
            print('  GME: x-> {}  y-> {} sum -> {}'.format(gmex,gmey,gmexy))
            errx, erry = self.fitParameters()
            self.actualizeModel(errx,erry)
            gme.append(self.getGlobalMachineError())
            self.computedOrmdP(list(self.kickers.keys()),self.dKickers)
            self.initializeAMat()
            gmex_new, gmey_new = self.getGlobalMachineError()
            converged = (abs(gmex_new + gmey_new - gmex_old - gmey_old) < error)
            it += 1

        np.save('errx', errx)
        np.save('erry', erry)
        np.save('dp0List', self.dp0List)
        np.save('dmonx',   self.dMonitors_x)
        np.save('dmony',   self.dMonitors_y)
        np.save('dkickers',self.dKickers)
        self._displayResultsNotSimple(it, gme, converged, plot=True)
        self._plotFit(modelx0,modely0)

    def setpList(self, pList):
        self.pList   = pList
        self.nParams = len(pList)
        self.dp0List = np.zeros(self.nParams)

    def computedOrmdP(self, kickers=[], dKickers=[]):
        """
        Computes the orbit response matrix derivative
        """
        dOrmx = []
        dOrmy = []
        #Initialize dp0 with zeros
        if len(self.dp0List) == 0: self.dp0List = np.zeros(len(self.pList))
        for m in range(len(self.messFiles)):
            measurement = self.messFiles[m]
            dmonx = self.dMonitors_x[m]
            dmony = self.dMonitors_y[m]
            ormA = self.orbitResponseAnalyzer
            ormA.setData(measurement)
            dormx, dormy = ormA.centralDorm(self.pList, self.dp0List,
                                            kickers, dKickers,
                                            dmonx, dmony)
            dOrmx.append(dormx)
            dOrmy.append(dormy)
        dormx = []
        dormy = []
        for j in range(self.nParams):
            myArrax = []
            myArray = []
            for i in range(len(self.messFiles)):
                for k in dOrmx[i][j]:
                    myArrax.append(k)
                for k in dOrmy[i][j]:
                    myArray.append(k)
            dormx.append(myArrax)
            dormy.append(myArray)
        self.dOrmxdP = dormx
        self.dOrmydP = dormy

    def _displayResults(self, kickerErrFit, monxErrFit, monyErrFit,
                        it, gme, converged, plot=True):
        gme = np.transpose(gme)
        if(plot):
            plt.plot(abs(gme[0][1:]-gme[0][:-1]),marker='.',label=r'$\zeta_x$')
            plt.plot(abs(gme[1][1:]-gme[1][:-1]),marker='.',label=r'$\zeta_y$')
            plt.xlabel('Iteration')
            plt.ylabel(r'|$\zeta_{i+1}$ - $\zeta_{i}$|)')
            plt.yscale('log')
            plt.legend()
            plt.show()
        kickerErrFit = np.transpose(kickerErrFit)
        relFit    = np.ones(self.nKickers)
        for k in range(self.nKickers):
            for c in kickerErrFit[k]:
                relFit[k] *= (1 + c)
        labls = list(self.kickers.keys())
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        if(converged):
            print('            Fit converged')
        else:
            print('      Fit didnt converged')
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('Number of iterations: ', it)
        print('')
        print('Initial GME: x-> ', round(gme[0][0],3), ' y-> ', round(gme[1][0],3))
        print('')
        print('Final   GME: x-> ', round(gme[0][-1],3),' y-> ', round(gme[1][-1],3))
        print('')
        print('Singular value threshold: ', self.singularMask )
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        print('Fit parameters for kickers:')
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        for k in range(self.nKickers):
            fit_k = (relFit[k]-1)*100
            print('{} : {} %'.format(labls[k], round(fit_k,3)))
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        print('Fit parameters for monitors')
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        monxErrFit = np.transpose(monxErrFit)
        monyErrFit = np.transpose(monyErrFit)
        relFitx = np.ones(self.nMonitors)
        relFity = np.ones(self.nMonitors)
        for m in range(self.nMonitors):
            for c in monxErrFit[m]:
                relFitx[m] *= (1 + c)
            for c in monyErrFit[m]:
                relFity[m] *= (1 + c)
        for m in range(self.nMonitors):
            fitx_m = (relFitx[m]-1)*100
            fity_m = (relFity[m]-1)*100
            print('{} : x-> {} %  y-> {} %'.format(self.monitors[m],
                                                 round(fitx_m,3),
                                                 round(fity_m,3)))
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')

    def _displayResultsNotSimple(self, it, gme, converged, plot=True):
        gme = np.transpose(gme)
        if(plot):
            plt.plot(abs(gme[0][1:]-gme[0][:-1]),marker='.',label=r'$\zeta_x$')
            plt.plot(abs(gme[1][1:]-gme[1][:-1]),marker='.',label=r'$\zeta_y$')
            plt.xlabel('Iteration',fontsize=12)
            plt.ylabel(r'|$\zeta_{i+1}$ - $\zeta_{i}$|',fontsize=14)
            plt.yscale('log')
            plt.legend()
            plt.show()
        labls = list(self.kickers.keys())
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        if(converged):
            print('            Fit converged')
        else:
            print('      Fit didnt converged')
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('Number of iterations: ', it)
        print('')
        print('Initial GME: x-> ', round(gme[0][0],2),
              ' y-> ',  round(gme[1][0],2) ,
              ' sum-> ',round(gme[0][0]+gme[1][0],2))
        print('')
        print('Final   GME: x-> ', round(gme[0][-1],2),
              ' y-> ', round(gme[1][-1],2),
              'sum->', round(gme[1][-1]+gme[0][-1],2))
        print('')
        print('Singular value threshold: ', self.singularMask )
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        print('Fit parameters for kickers:')
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        for k in range(self.nKickers):
            fit_k = (self.dKickers[k]-1)*100
            print('{} : {} %'.format(labls[k], round(fit_k,3)))
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        print('Fit parameters for monitors')
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')

        for m in range(self.nMonitors):
            fitx_m = (self.dMonitors_x[m]-1)*100
            fity_m = (self.dMonitors_y[m]-1)*100
            print('{} : x-> {} %  y-> {} %'.format(self.monitors[m],
                                                 round(fitx_m,3),
                                                 round(fity_m,3)))
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        print('Fit parameters differences')
        print('')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('')
        ormA = self.orbitResponseAnalyzer
        madx = ormA.madx
        madx.call(file=ormA.madxModelFile, chdir=True)
        madx.globals.update(ormA.data['model'])
        for p in range(self.nParams):
            percent = (madx.globals[self.pList[p]] + self.dp0List[p]) / \
                      madx.globals[self.pList[p]]
            percent = (percent - 1)*100
            print('{} : {} %'.format(self.pList[p], round(percent,5)))

    def _plotFit(self, modelx0, modely0):
        kickers = list(self.kickers.keys())
        xlabels = []
        monPos  = []
        Mx      = np.transpose(self.ormMx)
        My      = np.transpose(self.ormMy)
        x = np.arange(len(Mx[2]))
        for i in range (len(Mx[1])):
            xlabels.append(kickers[int(Mx[1][i])])
            if (Mx[1][i] == 0): monPos.append(i)

        plt.figure(1)
        plt.errorbar(x,Mx[2],yerr=Mx[3],marker='.',label='Measured',
                     linestyle='', markersize=10)
        plt.plot(x,modelx0,label='Model',marker='', linewidth=2)
        plt.xlim(-0.2, x[-1]+0.25)
        plt.ylim(min(Mx[2])-2., max(Mx[2])+2.)
        plt.plot(x,Mx[4],label='Fit',marker='',linewidth=1.5)
        plt.ylabel(r'Horizontal orbit response [mm mrad$^{-1}$]',fontsize=14)
        plt.legend(loc=0)
        plt.tight_layout()
        for i in range(len(monPos)):
            plt.axvline(monPos[i],linestyle='--')
            plt.text(monPos[i]+0.2, min(Mx[2]), self.monitors[i],
                     rotation=90, alpha=0.5, fontsize=12)

        plt.figure(2)
        plt.errorbar(x,My[2],yerr=Mx[3],marker='.',label='Measured',
                     linestyle='',markersize=10)
        plt.plot(x,modely0,label='Model',marker='', linewidth=2)
        plt.xlim(-0.2, x[-1]+0.25)
        plt.ylim(min(My[2])-2., max(My[2])+2.)
        plt.plot(x,My[4],label='Fit',marker='', linewidth=1.5)
        plt.ylabel(r'Vertical orbit response [mm mrad$^{-1}$]',fontsize=14)
        plt.legend(loc=0)
        plt.tight_layout()
        locs, labels = plt.xticks()
        plt.xticks(x, xlabels, rotation='vertical', fontsize=10)
        plt.legend(loc=0)
        for i in range(len(monPos)):
            plt.axvline(monPos[i],linestyle='--')
            plt.text(monPos[i]+0.2, min(Mx[2]), self.monitors[i],
                     rotation=90, alpha=0.5, fontsize=12)
        plt.show()
        ormMx = self.ormMx
        ormMy = self.ormMy
        for i in range(len(self.ormMx)):
            ormMx[i]=np.append(ormMx[i], modelx0[i])
            ormMy[i]=np.append(ormMy[i], modely0[i])
        head = 'Monitor  Kicker   orm   dorm   madxOrm  fitOrm'
        np.savetxt('{}/ormx.txt'.format(self.savePath), ormMx,
                   fmt=['%i','%i','%e', '%e','%e','%e'],
                   delimiter='  ', header=head)
        np.savetxt('{}/ormy.txt'.format(self.savePath),
                   ormMy, fmt=['%i','%i','%e', '%e','%e','%e'],
                   delimiter='  ', header=head)
