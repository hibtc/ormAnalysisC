import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning
import pandas
import csv
warnings.simplefilter("error", OptimizeWarning)
warnings.simplefilter("error", RuntimeWarning)

class MonitorReader:
    
    def __init__(self, dataPath):
        """
        This class computes the Orbit Response of the beam with help
        of the grid profiles measured at each data acquisition session 
        """
        self.dataPath = dataPath
        
    def getMessInfo(self, messFile):
        with open(messFile, encoding='latin-1') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=';')
            data = []
            for row in csvReader: data.append(row)
            monitor = data[1][-1]
            time    = data[7][-1]
            time    = time.split(' ')
            time    = time[-1]
            mu0x    = float(data[21][-1])
            sig0x   = float(data[23][-1])
            mu0y    = float(data[26][-1])
            sig0y   = float(data[28][-1])
        
        return [monitor, time, mu0x, sig0x, mu0y, sig0y]
    
    def gaussCurve(self, x, a, mu, sig, l):
        G = a*np.exp(-(x-mu)**2/(2*sig**2)) + l
        return G

    def plotCurve(self, gitterFile):
        info = self.getMessInfo(gitterFile)
        data = np.loadtxt(gitterFile, delimiter=';',skiprows=34,
                          encoding='latin-1', unpack=True)
        # Position is the same for x and y
        posx = data[1]
        x    = data[3]
        y    = data[4]

        mux0, sigx0 = info[2], info[3]
        muy0, sigy0 = info[4], info[5]
        p0x = [max(x), mux0, sigx0, min(x)]
        p0y = [max(y), muy0, sigy0, min(y)]
        
        Gx, dGx = curve_fit(self.gaussCurve, posx, x, p0=p0x)
        Gy, dGy = curve_fit(self.gaussCurve, posx, y, p0=p0y)
        xfit = np.linspace(posx[0], posx[-1], 200)
        mux  = round(Gx[1],2)
        dmux = round(np.sqrt(dGx[1][1]),2)
        #dmux = 0
        muy  = round(Gy[1],2)
        dmuy = round(np.sqrt(dGy[1][1]),2)

        print('----------------------------')
        print('Monitor: {}'.format(info[0]))
        print('Time:    {}'.format(info[1]))
        print('----------------------------')
        print('mu_x: {}'.format(info[2]))
        print('Fit:  {} +- {}'.format(mux, dmux))
        print('mu_y: {}'.format(info[4]))
        print('Fit:  {} +- {}'.format(muy, dmuy))
        print('---------------------------')
        print(gitterFile)
        print('---------------------------')
        
        plt.figure(1)
        plt.plot(posx, x, marker='.', label='x', linestyle='')
        plt.plot(xfit, self.gaussCurve(xfit, *Gx))
        plt.xlabel('Position [mm]')
        plt.ylabel('Intensity [a.U.]')
        plt.legend(loc=0)
        
        plt.figure(2)
        plt.plot(posx, y, marker='.', label='y', linestyle='')
        plt.plot(xfit, self.gaussCurve(xfit, *Gy))
        plt.xlabel('Position [mm]')
        plt.ylabel('Intensity [a.U.]')
        plt.legend(loc=0)
        
        plt.show()

    def timeToMin(self, time):
        t = time.split(':')
        return float(t[0])*60+float(t[1])+float(t[2])/60

    def fitMonitor(self, monitorPath, showProfiles=False):
        
        gitterFiles = glob.glob((monitorPath+'/*'))
        positionFits = []
        #gitterFiles = gitterFiles[-15:]
        #print(gitterFiles[2])
        print('Anzahl von Files: {}'.format(len(gitterFiles)))
        for f in gitterFiles:
            info = self.getMessInfo(f)
            if showProfiles: self.plotCurve(f)
            data = np.loadtxt(f, delimiter=';', skiprows=34,
                              encoding='latin-1', unpack=True)
            monitor, time = info[0], info[1]
            time = self.timeToMin(time)
            
            mux0, sigx0 = info[2], info[3]
            muy0, sigy0 = info[4], info[5]
            posx = data[1]
            x    = data[3]
            y    = data[4]
            p0x = [max(x), mux0, sigx0, min(x)]
            p0y = [max(y), muy0, sigy0, min(y)]
            try:
                Gx, dGx = curve_fit(self.gaussCurve, posx, x, p0=p0x)
                Gy, dGy = curve_fit(self.gaussCurve, posx, y, p0=p0y)
                xfit = np.linspace(posx[0], posx[-1], 200)
                mux  = round(Gx[1],2)
                dmux = round(np.sqrt(dGx[1][1]),2)
                muy  = round(Gy[1],2)
                dmuy = round(np.sqrt(dGy[1][1]),2)
                if(dmux>30):
                    dmux=1e-2
                    mux=0.
                    print(f)
                if(dmuy>30):
                    dmuy=1e-2
                    muy=0.
                    print(f)
            except OptimizeWarning:
                print('Catching the exception')
                print(f)

            except RuntimeWarning:
                print(f)

            if(dmux > 5.): print(f)
            xMask = abs(mux-mux0)/dmux < 5.0
            yMask = abs(muy-muy0)/dmuy < 5.0
            if (xMask and yMask):
                positionFits.append([time, mux, dmux, muy, dmuy, mux0, muy0])

        self.plotFits(positionFits)
        #self.plotHistos(positionFits)

    def plotFits(self, positionFits):
        positionFits = np.transpose(positionFits)
        t = positionFits[0]
        mux  = positionFits[1]
        dmux = positionFits[2]
        muy  = positionFits[3]
        dmuy = positionFits[4]
        mux0 = positionFits[5]
        muy0 = positionFits[6]
        
        plt.figure(1)
        plt.errorbar(t, mux, yerr=dmux, marker='.')
        plt.plot(t, mux0, marker='.', linestyle='')
        plt.title('x-Position')
        plt.xlabel('Time')
        plt.ylabel('Position')
        
        plt.figure(2)
        plt.errorbar(t, muy, yerr=dmuy, marker='.')
        plt.plot(t, muy0, marker='.', linestyle='')
        plt.title('y-Position')
        plt.xlabel('Time')
        plt.ylabel('Position')

        plt.show()

    def plotHistos(self, positionFits):

        positionFits = np.transpose(positionFits)
        t = positionFits[0]
        mux  = positionFits[1]
        dmux = positionFits[2]
        muy  = positionFits[3]
        dmuy = positionFits[4]

        plt.figure(1)
        plt.hist(mux,bins=15)
        plt.title(r'$\mu_x$')
        plt.figure(2)
        plt.hist(dmux,bins=10)
        plt.title(r'$\Delta \mu_x$')
        plt.figure(3)
        plt.hist(muy,bins=15)
        plt.title(r'$\mu_y$')
        plt.figure(4)
        plt.hist(dmuy,bins=10)
        plt.title(r'$\Delta \mu_y$')

        plt.show()

def testGitterReader():
    gitterPath = '/home/cristopher/HIT/ormData/ormMessdata/10-06-2019/GitterProfile/'
    gitterProfile = 'ProfilePT2/'
    monitorsT1 = ['H1DG1G', 'H1DG2G', 'B1DG2G', 'B1DG3G']
    monitorsT2 = ['H1DG1G', 'H1DG2G', 'H2DG2G', 'B2DG2G', 'B2DG3G']
    monitorsT3 = ['H1DG1G', 'H1DG2G', 'H2DG2G', 'H3DG3G',
                  'B3DG2G', 'B3DG3G', 'G3DG3G', 'G3DG5G', ]
    
    gR = MonitorReader(gitterPath)
    #gR.plotCurve(messFile)
    for m in monitorsT2:
        print(m)
        messung = gitterPath + gitterProfile + m
        gR.fitMonitor(messung,
                      showProfiles=0)
    
testGitterReader()
