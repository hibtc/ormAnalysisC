import OrmAnalysis as ormAna
import numpy as np
import matplotlib.pyplot as plt

class OrbitResponseMatrix:

    def __init__(self, messFiles, monitors, madxFile, readOrm = False):

        self.messFiles = messFiles
        self.monitors  = monitors
        self.madxFile  = madxFile
        self.kickers   = {}
        self.nMonitors = len(monitors)
        """
        We can spare some time if the ORM has already been computed
        to tweak the numerics, note some numerical precision is lost
        but results will still look qualitatively the same
        """
        if(readOrm):
            self.ormMx, self.ormMy = self.readMessOrm()
            self.setKickers()
        else:
            self.ormMx, self.ormMy = self.writeMessOrm()
        self.nKickers = len(self.kickers)
        self.Mx = 0
        self.My = 0
        self.dp0 = np.zeros(self.nMonitors + self.nKickers)
        self.computeOrm()

    def writeMessOrm(self, write=True):
        """
        Computes the measured orbit response and writes a text file
        which contains the numbering of the monitor, the numbering of
        the orbit corrector and the measured value.
        """
        ormx = []
        ormy = []

        for measurement in self.messFiles:

            ormA = ormAna.OrmAnalysis( measurement, self.madxFile )
            ks =  ormA.kickers
            m  =  ormA.monitor
            for k in ks:
                if k not in self.kickers: self.kickers[k] = len(self.kickers) + 1

            mMeasured, dmMeasured = ormA.computeOrmMeasured()
            mModelx, mModely = ormA.computeOrmModel()
            mMx = mMeasured[0]
            mMy = mMeasured[1]
            dmMx = dmMeasured[0]
            dmMy = dmMeasured[1]

            for i in range(len(ks)):
                ormx.append([self.monitors[m], i, mMx[i], dmMx[i], mModelx[i]])
                ormy.append([self.monitors[m], i, mMy[i], dmMy[i], mModely[i]])

            """
            #Simulated data --> Algorithm works alright!!!
            for i in range(len(ks)):
                if i == 2 or i ==3:
                    ormx.append([self.monitors[m], i, mModelx[i]*1.1, 1e-3, mModelx[i]])
                    ormy.append([self.monitors[m], i, mModely[i]*1, dmMy[i], mModely[i]])
                else:
                    r = 1 + 1*np.random.rand()/100
                    ormx.append([self.monitors[m], i, mModelx[i]*r, 1e-3, mModelx[i]])
                    ormy.append([self.monitors[m], i, mModely[i], dmMy[i], mModely[i]])
            """
        if(write):
            head = 'Monitor  Kicker   orm   dorm   madxOrm'
            np.savetxt('ormx.txt',ormx, fmt=['%i','%i','%e', '%e','%e'],
                       delimiter='  ', header=head)
            np.savetxt('ormy.txt',ormy, fmt=['%i','%i','%e', '%e','%e'],
                       delimiter='  ', header=head)
        return ormx, ormy

    def setKickers(self):
        """
        This is for the case where we just read the data. We build a dictionary
        of kickers from the last monitor.
        """
        ormA = ormAna.OrmAnalysis( self.messFiles[-1], self.madxFile )
        ks =  ormA.kickers
        for k in ks:
            if k not in self.kickers: self.kickers[k] = len(self.kickers) + 1

    def readMessOrm(self):
        """
        Read the ORM already computed
        """
        ormx = np.loadtxt('ormx.txt')
        ormy = np.loadtxt('ormy.txt')
        return ormx, ormy

    def computeOrm(self, visualize=False):
        """
        Orders the values as in Appendix A of the Overleaf PDF.
        This just takes into consideration the monitors and the
        kickers, the derivative of the ORM has still to be implemented.
        """
        nColumns = len(self.ormMx)
        nRows = self.nMonitors + self.nKickers
        Mx = np.zeros((nColumns, nRows))
        My = np.zeros((nColumns, nRows))

        for i in range(nColumns):

            Cijx = self.ormMx[i]
            Cijy = self.ormMy[i]
            entryCijx = Cijx[4] / Cijx[3]
            entryCijy = Cijy[4] / Cijy[3]
            Mx[i][int(Cijx[0])] = -entryCijx
            Mx[i][self.nMonitors + int(Cijx[1])] = entryCijx
            My[i][int(Cijy[0])] = -entryCijy
            My[i][self.nMonitors + int(Cijy[1])] = entryCijy

        """
        In case we want to see that everything works fine.
        Difficult to watch though, since we have an ~50x(7+19)
        matrix.
        """
        if(visualize):
            for i in My:
                for j in i:
                    print(round(j,4), end='  ')
                print()

        self.Mx = Mx
        self.My = My

    def fitParameters(self, singularMask=1e-4):
        """
        Here the parameters are fitted as in
        XFEL Beam Dynamics meeting (12.12.2005)
        but without the derivate part of the ORM.
        @param singularMask controls the entries that will be taken into
               account in the singular value decomposition.
        """
        vecOrmx = self.ormMx
        vecOrmy = self.ormMy

        vecOrmx = np.transpose(vecOrmx)
        vecOrmy = np.transpose(vecOrmy)

        vecOrmx = (vecOrmx[2]-vecOrmx[4])/vecOrmx[3]
        vecOrmy = (vecOrmy[2]-vecOrmy[4])/vecOrmy[3]

        # Singular Value Decomposition
        # dp = (B+)*A^{T}*(ORM_{gemessen} - ORM_{model})
        # B  = A^{T}*A
        # B+ = V*D*U^{T} from SVD
        #
        # See XFEL Beam Dynamics meeting (12.12.2005):
        # 'Response matrix measurements and
        #  analysis at DESY'

        M = np.vstack((self.Mx, self.My))
        Mt = np.transpose(M)
        B = np.matmul(Mt,M)

        U,S,Vt = np.linalg.svd( B, full_matrices=False, compute_uv=True)
        D = 1/S

        for s in range(len(D)):
            if D[s] > singularMask: D[s] = 0.

        D = np.diag(D)
        V = np.transpose(Vt)
        Ut =  np.transpose(U)
        Bplus = np.matmul(V,np.matmul(D,Ut))
        y = np.hstack((vecOrmx, vecOrmy))
        dp0 = np.matmul(Bplus, Mt)
        self.dp0 = np.matmul(dp0, y)

        """
        We scale the monitor reads and the kickers according to
        Eqs. 8 and 9.
        """
        vecOrmx = self.ormMx
        vecOrmy = self.ormMy

        for c in vecOrmx:
            f =( ( 1 + self.dp0[self.nMonitors + int(c[1])] ) /
                 ( 1 + self.dp0[int(c[0])] ) )
            c[4] *= f

        for c in vecOrmy:
            f =( ( 1 + self.dp0[self.nMonitors + int(c[1])] ) /
                 ( 1 + self.dp0[int(c[0])] ) )
            c[4] *= f

        self.ormMx = vecOrmx
        self.ormMy = vecOrmy
        self.computeOrm()

        L2error = np.sqrt(sum(y)**2)

        return L2error

    def optimize(self, nIterations=100,plot=True, singularMask=1e-4):
        """
        Computes the optimization scheme with help of the
        function fitParameters and is able to plot the results.
        It prints out the computed fit parameters for the corrector
        kickers and the monitors.
        @param nIterations is the number of iterations the algorithm
               will run.
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

        for i in range( nIterations ):
            monx = self.dp0[:self.nMonitors]
            kickx = self.dp0[self.nMonitors:]
            kickerFits.append(kickx)
            monFits.append(monx)
            error = self.fitParameters(singularMask)
            L2.append(error)

        kickerFits = np.transpose(kickerFits)

        for i in range(len(kickerFits)):
            print(kickNames[i],' : ',round(sum(kickerFits[i]),4))

        monFits = np.transpose(monFits)
        for i in range(len(monFits)):
            print(monNames[i]," : " ,round(sum(monFits[i]),4))

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
            plt.ylabel(r'$||\vec{y}||_2$ (See Eq. 11 in Appendix)')
            #plt.yscale('log')
            plt.show()

def main():
    # Here are the measurements of the first session
    prePath1 ="../ormAnalysis/ormMessdata/2018-10-20-orm_measurements/M8-E108-F1-I9-G1/"
    messFiles1 = [
        '2018-10-21_04-23-18_hht3_h1dg1g.orm_measurement.yml',
        '2018-10-21_04-16-30_hht3_h1dg2g.orm_measurement.yml',
        '2018-10-21_04-08-39_hht3_h2dg2g.orm_measurement.yml',
        '2018-10-21_03-54-09_hht3_h3dg3g.orm_measurement.yml',
        '2018-10-21_03-38-51_hht3_b3dg2g.orm_measurement.yml',
        '2018-10-21_03-21-09_hht3_b3dg3g.orm_measurement.yml',
        #'2018-10-21_02-50-02_hht3_g3dg3g.orm_measurement.yml',
        #'2018-10-21_02-25-45_hht3_g3dg5g.orm_measurement.yml',
        #'2018-10-21_01-52-39_hht3_t3df1.orm_measurement.yml',
    ]

    prePath3 = '../ormAnalysis/ormMessdata/2019-05-11/ORM_Daten/'
    # Transfer Line Daten ohne Gantry for the second session
    messFiles3 = [
        '2019-05-12_02-17-11_hht2_h1dg1g.orm_measurement.yml',
        '2019-05-12_02-21-21_hht2_h1dg2g.orm_measurement.yml',
        '2019-05-12_02-27-35_hht2_h2dg2g.orm_measurement.yml',
        '2019-05-12_02-39-53_hht3_h3dg3g.orm_measurement.yml',
        '2019-05-12_02-51-13_hht3_b3dg2g.orm_measurement.yml',
        '2019-05-12_03-05-19_hht3_b3dg3g.orm_measurement.yml',
    ]

    for f in range(len(messFiles1)):
        messFiles1[f] = prePath1 + messFiles1[f]
        messFiles3[f] = prePath3 + messFiles3[f]

    monitors = {'h1dg1g':0, 'h1dg2g':1, 'h2dg2g':2, 'h3dg3g':3,'b3dg2g':4, 'b3dg3g':5}#,'g3dg3g':6,'g3dg5g':7,'t3df1':8}
    madxFile = "../ormAnalysis/hit_models/hht3/run.madx"

    orm = OrbitResponseMatrix( messFiles3, monitors, madxFile, readOrm=1 )
    orm.optimize(nIterations=10, plot=1, singularMask=1e2)

main()
