import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)

class MonitorReader:
    
    def __init__(self, dataPath):
        """
        This class computes the Orbit Response of the beam with help
        of the grid profiles measured at each data acquisition session 
        """
        self.dataPath = dataPath
        
    def getMessInfo(self, messFile):
        with open(messFile, encoding='latin-1') as gridProfile:
            f = gridProfile.readlines()
            monitor = f[1]
            time    = f[7]
            mu0x    = f[21]
            sig0x   = f[23]
            mu0y    = f[26]
            sig0y   = f[28]
            monitor = monitor[len(monitor)-7:len(monitor)]
            time    = time[len(time)-9:len(time)]
            mu0x    = float(mu0x[len(mu0x)-7:len(mu0x)])
            mu0y    = float(mu0y[len(mu0y)-7:len(mu0y)])
            sig0x   = float(sig0x[len(sig0x)-7:len(sig0x)])
            sig0y   = float(sig0y[len(sig0y)-7:len(sig0y)])

        return [monitor.replace('\n',''), time.replace('\n',''),
                mu0x, sig0x, mu0y, sig0y]

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

        mux0, sigx0 = -info[2], info[3]
        muy0, sigy0 = info[4], info[5]
        p0x = [max(x), mux0, sigx0, min(x)]
        p0y = [max(y), muy0, sigy0, min(y)]
        
        Gx, dGx = curve_fit(self.gaussCurve, posx, x, p0=p0x)
        Gy, dGy = curve_fit(self.gaussCurve, posx, y, p0=p0y)
        xfit = np.linspace(posx[0], posx[-1], 200)
        mux  = round(Gx[1],2)
        dmux = round(np.sqrt(dGx[1][1]),2)
        muy  = round(Gy[1],2)
        dmuy = round(np.sqrt(dGy[1][1]),2)

        print('----------------------------')
        print('Monitor: {}'.format(info[0]))
        print('Time:    {}'.format(info[1]))
        print('----------------------------')
        print('mu_x: {}'.format(info[2]))
        print('Fit:  {} +- {}'.format(-mux, dmux))
        print('mu_y: {}'.format(info[4]))
        print('Fit:  {} +- {}'.format(muy, dmuy))
        
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

        print('Anzahl von Files: {}'.format(len(gitterFiles)))
        for f in gitterFiles:
            info = self.getMessInfo(f)
            if showProfiles: self.plotCurve(f)
            data = np.loadtxt(f, delimiter=';', skiprows=34,
                              encoding='latin-1', unpack=True)
            monitor, time = info[0], info[1]
            time = self.timeToMin(time)
            
            mux0, sigx0 = -info[2], info[3]
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
                if(dmux>30): dmux=0.
                if(dmuy>30): dmuy=0.
                    
            except OptimizeWarning:
                print('Catching the exception')
                print(f)


            if(dmux > 5.): print(f)
            positionFits.append([time, mux, dmux, muy, dmuy, mux0, muy0])

        self.plotFits(positionFits)
        self.plotHistos(positionFits)

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
    gitterProfile = 'ProfilePT2/H1DG1G'
    monitors = ['H1DG1G','H1DG2G', 'B1DG2G', 'B1DG3G']
    messung = gitterPath + gitterProfile  

    gR = MonitorReader(gitterPath)
    #messFile = '/home/cristopher/HIT/ormData/ormMessdata/10-06-2019/GitterProfile/ProfilePT1/B1DG2G/B1DG2G_6_20518976_1.CSV'
    #gR.plotCurve(messFile)
    gR.fitMonitor(messung,
                  showProfiles=False)
    
testGitterReader()
