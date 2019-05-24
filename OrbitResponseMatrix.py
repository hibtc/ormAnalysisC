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
        if(readOrm):
            self.ormMx, self.ormMy = self.readMessOrm()
            self.setKickers()
        else:
            self.ormMx, self.ormMy = self.writeMessOrm()
        self.nKickers = len(self.kickers)
        self.Mx = 0
        self.My = 0
        self.dp0x = np.zeros(self.nMonitors + self.nKickers)
        self.dp0y = np.zeros(self.nMonitors + self.nKickers)

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
            np.savetxt('ormx.txt',ormx, fmt=['%i','%i','%f16', '%f16','%f16'],
                       delimiter='  ', header=head)
            np.savetxt('ormy.txt',ormy, fmt=['%i','%i','%f16', '%f16','%f16'],
                       delimiter='  ', header=head)
        return ormx, ormy

    def setKickers(self):
        ormA = ormAna.OrmAnalysis( self.messFiles[-1], self.madxFile )
        ks =  ormA.kickers
        for k in ks:
            if k not in self.kickers: self.kickers[k] = len(self.kickers) + 1

    def readMessOrm(self):

        ormx = np.loadtxt('ormx.txt')
        ormy = np.loadtxt('ormy.txt')
        return ormx, ormy

    def computeOrm(self, visualize=False):

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

        if(visualize):
            for i in My:
                for j in i:
                    print(round(j,4), end='  ')
                print()

        self.Mx = Mx
        self.My = My

    def fitParameters(self):

        vecOrmx = self.ormMx
        vecOrmy = self.ormMy
        # Here we scalate the model with the fitted parameters
        #dp0 = (self.dp0x + self.dp0y)/2
        #print(self.dp0x[self.nMonitors:])

        for c in vecOrmx:
            f =( ( 1 + self.dp0x[int(c[0])] ) /
                 ( 1 + self.dp0x[self.nMonitors + int(c[1])] ) )
            #if abs(f) < 1.4:
            c[4] *= f

        # Here we scalate the model with the fitted parameters
        for c in vecOrmy:
            f =( ( 1 + self.dp0x[int(c[0])] ) /
                 ( 1 + self.dp0x[self.nMonitors + int(c[1])] ) )
            #if abs(f) < 1.1:
            c[4] *= f

        self.ormMx = vecOrmx
        self.ormMy = vecOrmy
        self.computeOrm()

        vecOrmx = np.transpose(vecOrmx)
        vecOrmx = (vecOrmx[2]-vecOrmx[4])/vecOrmx[3]

        vecOrmy = np.transpose(vecOrmy)
        vecOrmy = (vecOrmy[2]-vecOrmy[4])/vecOrmy[3]

        for c in range(len(vecOrmx)):
            if vecOrmx[c] > 1: vecOrmx[c] = 0.

        for c in range(len(vecOrmy)):
            if vecOrmy[c] > 1: vecOrmy[c] = 0.

        dp0x = np.zeros(self.nMonitors +  self.nKickers)
        dp0y = np.zeros(self.nMonitors +  self.nKickers)

        # Singular Value Decomposition
        # dp = (B+)*A^{T}*(ORM_{gemessen} - ORM_{model})
        # B  = A^{T}*A
        # B+ = V*D*U^{T} from SVD
        #
        # See XFEL Beam Dynamics meeting (12.12.2005):
        # 'Response matrix measurements and
        #  analysis at DESY'

        Ux,Sx,Vx = np.linalg.svd( self.Mx, full_matrices=False, compute_uv=True)
        Uy,Sy,Vy = np.linalg.svd( self.My, full_matrices=False, compute_uv=True)

        for i in range(len(Sx)):
            if Sx[i] == 0: Sx[i] = 0.01
            if Sy[i] == 0: Sy[i] = 0.01

        Dx = (1/Sx)
        Dy = (1/Sy)

        for i in range(len(Dx)):
            if Dx[i] > 0.03 : Dx[i] = 0.
            if Dy[i] > 0.03 : Dy[i] = 0.

        Dx = np.diag(Dx)
        Amx = np.matmul(Vx,np.matmul(Dx,np.transpose(Ux)))
        dp1x = np.matmul(Amx, vecOrmx)
        #L2error = np.sqrt(sum((dp1-dp0)**2))
        Dy = np.diag(Dy)
        Amy = np.matmul(Vy,np.matmul(Dy,np.transpose(Uy)))
        dp1y = np.matmul(Amy, vecOrmy)

        self.dp0x = dp1x
        self.dp0y = dp1y
        L2error = np.sqrt(sum((self.dp0x)**2))
        L2error = np.sqrt(sum(vecOrmx)**2)
        return L2error


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

    monitors = {'h1dg1g':0, 'h1dg2g':1, 'h2dg2g':2, 'h3dg3g':3,'b3dg2g':4, 'b3dg3g':5  }# ,'g3dg3g':6,'g3dg5g':7,'t3df1':8}
    madxFile = "../ormAnalysis/hit_models/hht3/run.madx"

    orm = OrbitResponseMatrix( messFiles1, monitors, madxFile, readOrm=False )
    orm.computeOrm()
    L2 = []

    Mx = np.transpose(orm.ormMx)
    My = np.transpose(orm.ormMy)

    x = np.linspace(0,len(Mx[2]),len(Mx[2]))
    plt.figure(1)
    plt.errorbar(x,Mx[2],yerr=Mx[3],marker='.',label='xMeasured')
    plt.plot(x,Mx[4],label='xModel',marker='.')
    plt.legend(loc=0)
    plt.figure(2)
    plt.errorbar(x,My[2],yerr=Mx[3],marker='.',label='yMeasured')
    plt.plot(x,My[4],label='yModel',marker='.')
    plt.legend(loc=0)

    kickerFits = []
    monFits = []
    kickNames = list(orm.kickers.keys())
    # For Messung 1 gibt es lokale Minimum bei 14 iterationen
    #for i in range(14):
    for i in range(16):

        monx = orm.dp0x[:orm.nMonitors]
        kickx = orm.dp0x[orm.nMonitors:]

        kickerFits.append(kickx)
        monFits.append(monx)
        error = orm.fitParameters()
        L2.append(error)

    #np.savetxt('kickerFit.txt', kickerFits)
    kickerFits = np.transpose(kickerFits)

    for i in range(len(kickerFits)):
        print(kickNames[i],' x: ',round(sum(kickerFits[i]),4))

    monFits = np.transpose(monFits)
    for m in monFits:
        print('m:', sum(m))

    orm.fitParameters()
    Mx = np.transpose(orm.ormMx)
    My = np.transpose(orm.ormMy)

    plt.figure(1)
    plt.plot(x,Mx[4],label='xFit',marker='.')
    plt.legend(loc=0)
    plt.figure(2)
    plt.plot(x,My[4],label='yFit',marker='.')
    plt.legend(loc=0)

    plt.show()

    print(min(L2))
    plt.plot(L2,marker='.')
    plt.yscale('log')
    plt.show()

main()
