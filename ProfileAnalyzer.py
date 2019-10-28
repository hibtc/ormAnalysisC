import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
import csv
from yaml import safe_load
warnings.simplefilter("error", OptimizeWarning)
warnings.simplefilter("error", RuntimeWarning)

class ProfileAnalyzer:

    def __init__(self, madguiData, monitorPath):
        """
        This class computes the Orbit Response of the beam with help
        of the grid profiles measured at each data acquisition session
        @param madguiData is the unpacked measurement
        @param monitorPath the path to the monitor profile files
        """
        self.madguiData  = madguiData
        self.monitorPath = monitorPath
        self.messDatax   = {}
        self.messDatay   = {}

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

    def doppelGauss(self, x, a1, a2, mu1, mu2, sig1, sig2, l):
        doppelG = a1*np.exp(-(x-mu1)**2/(2*sig1**2)) + \
                  a2*np.exp(-(x-mu2)**2/(2*sig2**2)) + l
        return doppelG

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
        plt.plot(posx, x, marker='.', label='Data', linestyle='')
        plt.plot(xfit, self.gaussCurve(xfit, *Gx), label='Fit')
        plt.xlabel('x Position [mm]')
        plt.ylabel('Intensity [a.U.]')
        plt.legend(loc=0)

        plt.figure(2)
        plt.plot(posx, y, marker='.', label='Data', linestyle='')
        plt.plot(xfit, self.gaussCurve(xfit, *Gy), label='Fit')
        plt.xlabel('y Position [mm]')
        plt.ylabel('Intensity [a.U.]')
        plt.legend(loc=0)

        plt.show()

    def timeToMin(self, time):
        t = time.split(':')
        return float(t[0])*60+float(t[1])+float(t[2])/60

    def fitProfiles(self, monitor, showProfiles=False,
                          skipShots=1, plot=True):
        """
        Fits the monitor profiles with a Gauss peak
        """
        gitterFiles = glob.glob((self.monitorPath+monitor+'/*'))
        positionFits = []

        print('------------------------------')
        print('  Fitting monitor profiles ')
        print('- Monitor: {}'.format(monitor))
        print('- Number of Files: {}'.format(len(gitterFiles)))
        print('------------------------------')
            
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
                if (dmux > 30. or dmuy > 30.
                    or abs(mux) > 30.):
                    muy = 0.
                    mux = 0.
                    dmuy = 1e-6
                    dmux = 1e-6
            except OptimizeWarning:
                print('Catching the exception')
                print(f)

            except RuntimeWarning:
                print(f)

            positionFits.append([time, mux, dmux, muy, dmuy, mux0, muy0])

        if (len(gitterFiles) != 0):
            messung = self.getMeasurements(positionFits, skipShots)
            self.messDatax, self.messDatay = self.formatMeasurements(messung)
            if(plot):
                self.plotFits(positionFits, monitor, messung)
        else:
            self.messDatax, self.messDatay = {},{}

    def getMeasurements(self, positionFits, skipShots):
        """
        Computes the mean value of the fitted beam profiles
        A filter is provided to rule out abnormal values
        """
        positionFits = np.transpose(positionFits)
        t    = positionFits[0]
        mux  = positionFits[1]
        dmux = positionFits[2]
        muy  = positionFits[3]
        dmuy = positionFits[4]
        mux0 = positionFits[5]
        muy0 = positionFits[6]

        tMask, kickers = self.getTimeMask()
        messWertex  = []
        dMessWertex = []
        messWertey  = []
        dMessWertey = []
        for tRange in tMask:
            mask1 = tRange[0]*np.ones(len(t)) < t
            mask2 = t < tRange[1]*np.ones(len(t))
            mask  = mask1*mask2
            tOptik = t[mask]
            muxOptik = mux[mask]
            muxOptikFiltered = self._filterPeaks(muxOptik)
            messWertx, dmessWertx_syst, dmessWertx_stat = \
                                       self._computeMean(muxOptik, dmux, mask)
            muyOptik = muy[mask]
            muyOptikFiltered = self._filterPeaks(muyOptik)
            messWerty, dmessWerty_syst, dmessWerty_stat = \
                                       self._computeMean(muyOptik, dmuy, mask)
            messWertex.append(messWertx)
            dMessWertex.append([dmessWertx_syst,
                                dmessWertx_stat])
            messWertey.append(messWerty)
            dMessWertey.append([dmessWerty_syst,
                                dmessWerty_stat])

        return [messWertex, dMessWertex, messWertey, dMessWertey]

    def _computeMean(self, data, error, timeMask):
        mean = np.mean(data)
        error_syst = np.sqrt(sum(error[timeMask]**2))/len(error[timeMask])
        error_stat = np.std(data)
        return mean, error_syst, error_stat
        
    def _filterPeaks(self, data):
        filteredData = data
        if(len(data) > 3):
            # Taking out the first measurement
            # It is always buggy 
            filteredData = np.sort(data[1:])
            # Taking out the abnormal values
            filteredData = filteredData[1:-1]
        return filteredData            
        
    def plotFits(self, positionFits, monitor, messWerte):
        """
        Plots the computed mean values of the fitted peaks
        from the monitor profiles
        """
        positionFits = np.transpose(positionFits)
        t = positionFits[0]
        mux  = positionFits[1]
        dmux = positionFits[2]
        muy  = positionFits[3]
        dmuy = positionFits[4]
        mux0 = positionFits[5]
        muy0 = positionFits[6]

        messWertex  = messWerte[0]
        dMessWertex = messWerte[1]
        messWertey  = messWerte[2]
        dMessWertey = messWerte[3]

        tMask, kickers = self.getTimeMask()
        
        plt.figure(1)
        plt.errorbar(t, mux, yerr=dmux, marker='.', linestyle='')
        plt.plot(t, mux0, marker='.', linestyle='')
        plt.title('x-Position')
        plt.xlabel('Time [min]')
        plt.ylabel('Position [mm]')
        for i in range(len(messWertex)):
            label1 = '{} +- ({} + {})'.format(round(messWertex[i],3),
                                                  round(dMessWertex[i][0],3),
                                                  round(dMessWertex[i][1],3))
            plt.plot(tMask[i], [messWertex[i],messWertex[i]],
                     label=label1)
        for ti in tMask:
            plt.axvline(ti[0], linestyle='--')
            plt.axvline(ti[1], linestyle='--')
        plt.legend(loc=0)

        plt.figure(2)
        plt.errorbar(t, muy, yerr=dmuy, marker='.',linestyle='')
        plt.plot(t, muy0, marker='.', linestyle='')
        plt.title('y-Position at {}'.format(monitor))
        plt.xlabel('Time [min]')
        plt.ylabel('Position [mm]')
        for i in range(len(messWertey)):
            label1 = '{} +- ({} + {})'.format(round(messWertey[i],3),
                                              round(dMessWertey[i][0],3),
                                              round(dMessWertey[i][1],3))
            plt.plot(tMask[i], [messWertey[i],messWertey[i]],
                     label=label1)
        for ti in tMask:
            plt.axvline(ti[0], linestyle='--')
            plt.axvline(ti[1], linestyle='--')
        plt.legend(loc=0)
        plt.show()

    def formatMeasurements(self, messWerte):
        """
        Returns the computed measurements in a very nice format
         {'Kicker1':['x1 Value', 'x1 Uncertainty'],
          'Kicker2':['x2 Value', 'x2 Uncertainty'],
          ...]
        """
        ormMessx = {}
        ormMessy = {}
        # Conversion in [mm]
        messWertex  = np.array(messWerte[0])*1e-3
        dMessWertex = np.array(messWerte[1])*1e-3
        messWertey  = np.array(messWerte[2])*1e-3
        dMessWertey = np.array(messWerte[3])*1e-3
        
        tMask, kickers = self.getTimeMask()
        for i in range(len(messWertex)):
            kick = list(kickers[i].keys())
            messWertyi  = messWertey[i]
            dmessWertyi = dMessWertey[i][0] + dMessWertey[i][1]
            messWertxi  = messWertex[i]
            dmessWertxi = dMessWertex[i][0] + dMessWertex[i][1]
            if(len(kick)==0): kick = ''
            else: kick = kick[0]
            ormMessx.update( {kick:[messWertxi, dmessWertxi]} )
            ormMessy.update( {kick:[messWertyi, dmessWertyi]} )
        return ormMessx, ormMessy

    def getTimeMask(self):
        data = self.madguiData
        opticTime = []
        kickers   = []
        for kickerChange in data['records']:
            kickers.append(kickerChange['optics'])
            shotTimes = [kickerChange['time']]
            for shot in kickerChange['shots']:
                shotTimes.append(shot['time'])
            opticTime.append(shotTimes)
        timeMasks = []
        for change in opticTime:
            t0 = change[0].split(' ')
            t1 = change[-1].split(' ')
            timeMasks.append([self.timeToMin(t0[1]),
                              self.timeToMin(t1[1])])
        return timeMasks, kickers
