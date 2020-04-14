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
            linkeKante = float(data[24][-1])
            rechteKante = float(data[25][-1])

        return [monitor, time, mu0x, sig0x, mu0y, sig0y,
                linkeKante, rechteKante]

    def gaussCurve(self, x, a, mu, sig, l):
        G = a*np.exp(-(x-mu)**2/(2*sig**2)) + l
        return G

    def doppelGauss(self, x, mu1, mu2, sig1, sig2, a1, a2, l):
        doppelG = a1*np.exp(-(x-mu1)**2/(2*sig1**2)) + \
                  a2*np.exp(-(x-mu2)**2/(2*sig2**2)) + l
        return doppelG

    def plotCurve(self, gitterFile):

        info = self.getMessInfo(gitterFile)
        data = np.loadtxt(gitterFile, delimiter=';',skiprows=34,
                          encoding='latin-1', unpack=True)

        print('----------------------------')
        print('Monitor: {}'.format(info[0]))
        print('Time:    {}'.format(info[1]))
        print('----------------------------')
        print(gitterFile)
        print('---------------------------')
        #print('mu_x: {}'.format(info[2]))
        #print('mu_y: {}'.format(info[4]))
        # Position is the same for x and y
        posx = data[1]
        x    = data[3]
        y    = data[4]

        plt.figure(1)
        plt.plot(posx, x,
                 marker='.', markersize=8,
                 label='Wire readout', linestyle='')
        plt.xlabel('x Position [mm]')
        plt.ylabel('Intensity [a.U.]')

        plt.figure(2)
        plt.plot(posx, y,
                 marker='.', markersize=8,
                 label='Wire readout', linestyle='')
        plt.xlabel('y Position [mm]')
        plt.ylabel('Intensity [a.U.]')

        mux0, sigx0 = info[2], info[3]
        muy0, sigy0 = info[4], info[5]
        if(abs(sigx0) > 10.): sigx0 *= 0.1
        if(abs(sigy0) > 10.): sigy0 *= 0.1
        p0x = [max(x), mux0, sigx0, min(x)]
        p0y = [max(y), muy0, sigy0, min(y)]
        monitor = info[0]
        if (monitor == 'H3DG3G' or \
            monitor == 'B3DG2G' or \
            monitor == 'B3DG3G'):
            # Double Gauss fit
            p0 = [info[-2], info[-1], 2.0, 2.0, max(x), max(x), 0.]
            print(p0)
            Gx, dGx = curve_fit(self.doppelGauss, posx, x, p0=p0,
                                maxfev=5000)
            mux, dmux = self.getDoubleGauss(Gx, dGx)
            mux  = round(mux, 3)
            dmux = round(dmux, 3)
        else:
            Gx, dGx = curve_fit(self.gaussCurve, posx, x, p0=p0x,maxfev=5000)
            mux  = round(Gx[1],2)
            dmux = round(np.sqrt(dGx[1][1]),2)
            #dmux = 0
        Gy, dGy = curve_fit(self.gaussCurve, posx, y, p0=p0y,maxfev=5000)

        xfit = np.linspace(posx[0], posx[-1], 200)

        muy  = round(Gy[1],2)
        dmuy = round(np.sqrt(dGy[1][1]),2)

        print('mu_x: {}'.format(info[2]))
        print('Fit:  {} +- {}'.format(mux, dmux))
        print('mu_y: {}'.format(info[4]))
        print('Fit:  {} +- {}'.format(muy, dmuy))
        print('---------------------------')

        plt.figure(1)
        if (monitor == 'H3DG3G' or \
            monitor == 'B3DG2G' or \
            monitor == 'B3DG3G'):
            plt.plot(xfit, self.doppelGauss(xfit, *Gx),
                     linewidth=1.5,
                     linestyle='--',
                     label='Gauss fit')
        else:
            plt.plot(xfit, self.gaussCurve(xfit, *Gx),
                     linewidth=1.5,
                     linestyle='--',
                     label='Gauss fit')

        for i in posx: plt.axvline(i,color='gray',linewidth=0.6)
        plt.legend(loc=0)
        plt.tight_layout()


        plt.figure(2)
        plt.plot(xfit, self.gaussCurve(xfit, *Gy),
                 linewidth=1.5,
                 linestyle='--',
                 label='Gauss fit')

        for i in posx: plt.axvline(i,color='gray',linewidth=0.6)
        plt.tight_layout()
        plt.legend(loc=0)

        plt.show()

    def getDoubleGauss(self, Gx, dGx):
        mu1  = Gx[0]
        sig1 = np.sqrt(dGx[0][0])
        mu2  = Gx[1]
        sig2 = np.sqrt(dGx[3][3])
        A1  = Gx[4]
        dA1 = np.sqrt(dGx[4][4])
        A2  = Gx[5]
        dA2 = np.sqrt(dGx[5][5])
        w12 = A1+A2
        muW =  (A1*mu1 + A2*mu2) / (w12)
        dmuW = (((mu1-muW)**2)*(dA1**2) + ((mu2-muW)**2)*(dA2**2) + \
                (A1*sig1)**2 + (A2*sig2)**2)/ (w12**2)
        dmuW = np.sqrt(dmuW)
        return muW, dmuW

    def timeToMin(self, time):
        t = time.split(':')
        return float(t[0])*60+float(t[1])+float(t[2])/60

    def fitProfiles(self, monitor, showProfiles=False,
                    skipShots=1, plot=True):
        """
        Fits the monitor profiles with a Gauss peak
        """
        # Uncomment to plot beam profiles
        # showProfiles=True
        gitterFiles = glob.glob((self.monitorPath+monitor+'/*'))
        gitterFiles.sort()
        positionFits = []
        envelopeFits = []

        print('------------------------------')
        print('  Fitting monitor profiles ')
        print('- Monitor: {}'.format(monitor))
        print('- Number of Files: {}'.format(len(gitterFiles)))
        print('------------------------------')

        for f in gitterFiles:
            info = self.getMessInfo(f)
            sigx0 = info[3]
            if sigx0 == -9999.0:
                print(f)
            #if monitor == 'G3DG3G':
            #    self.handFitG3DG3G(f)

        for f in gitterFiles:
            info = self.getMessInfo(f)
            if showProfiles: self.plotCurve(f)
            data = np.loadtxt(f, delimiter=';', skiprows=34,
                              encoding='latin-1', unpack=True)
            monitor, time = info[0], info[1]

            mux0, sigx0 = info[2], info[3]
            muy0, sigy0 = info[4], info[5]

            envx0 = sigx0
            envy0 = sigy0

            time = self.timeToMin(time)

            if(time < self.timeToMin('14:21:00') and
               time > self.timeToMin('14:16:45') and
               monitor == 'G3DG3G'):
               #print(time)
               muy0 = -21.

            posx = data[1]
            x    = data[3]
            y    = data[4]

            if(sigx0 > 10.): sigx0 *= 0.1
            if(sigy0 > 10.): sigy0 *= 0.1
            p0x = [max(x), mux0, sigx0, min(x)]
            p0y = [max(y), muy0, sigy0, min(y)]

            #if (monitor == 'G3DG3G'):
            #    if time < self.timeToMin('20:27:46'):
            #        p0x = [max(x),  0.,  1., min(x)]
            #        p0y = [max(y), -7.6, 3., min(y)]
            #    elif (time < self.timeToMin('20:42:18') and
            #          time >= self.timeToMin('20:37:35')):
            #        p0x = [max(x), -1.,  1.0, min(x)]
            #        p0y = [max(y), -23., 3.0, min(y)]
            #    elif (time < self.timeToMin('20:45:04') and
            #          time >= self.timeToMin('20:44:15')):
            #        p0x = [max(x), 0.,  1., min(x)]
            #        p0y = [max(y), 0.,  3., min(y)]
            #     else:
            #         p0x = [max(x), mux0,  sigx0, min(x)]
            #         p0y = [max(y), muy0,  sigy0, min(y)]

            try:
                if (monitor == 'H3DG3G'):
                    p0d = [info[-2], info[-1], 2.0, 2.0, max(x),0.6*max(x), 0.]
                    #print(p0d)
                    Gx, dGx = curve_fit(self.doppelGauss, posx, x, p0=p0d,
                                        maxfev=5000)
                    mux, dmux = self.getDoubleGauss(Gx, dGx)
                    envx = envx0
                    denvx = 0.

                if (monitor == 'B3DG2G' or \
                    monitor == 'B3DG3G'):
                    # Double Gauss fit
                    p0d = [info[-2], info[-1], 2.0, 2.0, 0.6*max(x), max(x), 0.]
                    #print(p0d)
                    Gx, dGx = curve_fit(self.doppelGauss, posx, x, p0=p0d,
                                        maxfev=5000)
                    mux, dmux = self.getDoubleGauss(Gx, dGx)
                    envx = envx0
                    denvx = 0.

                else:
                    Gx, dGx = curve_fit(self.gaussCurve, posx, x, p0=p0x,
                                        maxfev=5000)
                    mux  = Gx[1]
                    dmux = np.sqrt(dGx[1][1])
                    envx = Gx[2]
                    denvx = np.sqrt(dGx[2][2])

                Gy, dGy = curve_fit(self.gaussCurve, posx, y, p0=p0y)
                xfit = np.linspace(posx[0], posx[-1], 200)
                muy  = Gy[1]
                dmuy = np.sqrt(dGy[1][1])
                envy = Gy[2]
                denvy = np.sqrt(dGy[1][1])

                if (dmux > 30. or dmuy > 30.
                    or abs(mux) > 30.):
                    muy = 0.
                    mux = 0.
                    dmuy = 1e-6
                    dmux = -1
            except OptimizeWarning:
                print('Catching the exception')
                print(f)

            except RuntimeWarning:
                print(f)

            if(mux != 0. and dmux != -1):
                positionFits.append([time, mux, dmux, muy, dmuy, mux0, muy0])
                envelopeFits.append([time, envx, denvx, envy, denvy, envx0, envy0])

        if (len(gitterFiles) != 0):
            messung = self.getMeasurements(positionFits, skipShots)
            self.messDatax, self.messDatay = self.formatMeasurements(messung)
            envelopes = self.getMeasurements(envelopeFits, skipShots)
            self._printProfileSummary(self.messDatax, self.messDatay, envelopeFits)
            if(plot):
                self.plotFits(positionFits, monitor, messung)
                self.plotFits(envelopeFits, monitor, envelopes, titel='Beam envelope')
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
                                       self._computeMean(muxOptikFiltered, dmux, mask)
            muyOptik = muy[mask]
            muyOptikFiltered = self._filterPeaks(muyOptik)
            messWerty, dmessWerty_syst, dmessWerty_stat = \
                                       self._computeMean(muyOptikFiltered, dmuy, mask)
            messWertex.append(messWertx)
            dMessWertex.append([dmessWertx_syst,
                                dmessWertx_stat])
            messWertey.append(messWerty)
            dMessWertey.append([dmessWerty_syst,
                                dmessWerty_stat])

        return [messWertex, dMessWertex, messWertey, dMessWertey]

    def _computeMean(self, data, error, timeMask):
        if len(data)==0:
            return
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

    def plotFits(self, positionFits, monitor, messWerte,
                 titel=''):
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
        #####################################
        #     Static fit for H1DG1G         #
        # Result:                           #
        #  345 SHOTS                        #
        #  mux  = (-2.26 +- 0.016) mm       #
        #  muy  = (-0.28 +- 0.003) mm       #
        #  sigx = (0.410 +- 0.023) mm       #
        #  sigy = (0.059 +- 0.002) mm       #
        #####################################
        # sigx, sigy tell us the reproducibility of
        # initial conditions i.e. extraction offset

        # TODO: Move this to a dedicated function
        """
        avex  = np.average(mux)
        davex_stat = np.std(mux)
        davex_fit  = np.sqrt(sum(dmux**2))/len(dmux)
        avex = round(avex, 3)
        davex_stat = round(davex_stat, 4)
        davex_fit = round(davex_fit,  4)

        avey  = np.mean(muy)
        davey_stat = np.std(muy)
        davey_fit  = np.sqrt(sum(dmuy**2))/len(dmuy)
        avey = round(avey, 4)
        davey_stat = round(davey_stat, 4)
        davey_fit = round(davey_fit,  4)
        print('---------------')
        print('mux: {} +- {} + {}'.format(avex, davex_stat, davex_fit))
        print('muy: {} +- {}+ {}'.format(avey, davey_stat, davey_fit))
        print('---------------')
        input('Waiting...')

        plt.figure(3)
        #h = plt.hist(mux[:345], bins=60)
        h = plt.hist(mux, bins=6)
        x = (h[1][:-1] + h[1][1:])/2
        plt.plot(x, h[0])

        p0 = [10., 2.1, 0.1, 0.]
        Gx, dGx = curve_fit(self.gaussCurve, x, h[0], p0=p0 )
        print('A:   {} +- {}'.format(Gx[0], np.sqrt(dGx[0][0])))
        print('mu:  {} +- {}'.format(Gx[1], np.sqrt(dGx[1][1])))
        print('sig: {} +- {}'.format(Gx[2], np.sqrt(dGx[2][2])))
        print('l:   {} +- {}'.format(Gx[3], np.sqrt(dGx[3][3])))
        xPlot = np.linspace(x[0],x[-1], 200)
        plt.plot(xPlot, self.gaussCurve(xPlot,*Gx))
        plt.show()
        """
        plt.figure(1)
        plt.errorbar(t, mux, yerr=dmux, marker='.', linestyle='')
        #plt.plot(t, mux0, marker='.', linestyle='')
        #plt.title('x-Position')
        if(len(titel)): plt.title(titel)
        plt.xlabel('Time [min]', fontsize=10)
        plt.ylabel('x [mm]', fontsize=10)
        for i in range(len(messWertex)):
            label1 = '{} +- ({} + {})'.format(round(messWertex[i],3),
                                                  round(dMessWertex[i][0],3),
                                                  round(dMessWertex[i][1],3))
            plt.plot(tMask[i], [messWertex[i],messWertex[i]],linewidth=3,
                     label=label1)
        for ti in tMask:
            plt.axvline(ti[0], linestyle='--')
            plt.axvline(ti[1], linestyle='--')
        #plt.legend(loc=0)

        plt.figure(2)
        plt.errorbar(t, muy, yerr=dmuy, marker='.',linestyle='')
        #plt.plot(t, muy0, marker='.', linestyle='')
        #plt.title('y at {}'.format(monitor))
        if(len(titel)): plt.title(titel)
        plt.xlabel('Time [min]', fontsize=12)
        plt.ylabel('y [mm]',fontsize=12)
        for i in range(len(messWertey)):
            label1 = '{} +- ({} + {})'.format(round(messWertey[i],3),
                                              round(dMessWertey[i][0],3),
                                              round(dMessWertey[i][1],3))
            plt.plot(tMask[i], [messWertey[i],messWertey[i]],linewidth=3,
                     label=label1)
        for ti in tMask:
            plt.axvline(ti[0], linestyle='--')
            plt.axvline(ti[1], linestyle='--')
        #plt.legend(loc=0)
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

    def _printProfileSummary(self, posFitsx, posFitsy, envFits):

        print('-------------------------------')
        print('       Monitor summary')
        print('-------------------------------')
        print()
        print('--------  Offset [mm] ---------')
        print()
        x0 = posFitsx['']
        y0 = posFitsy['']
        print(' x -> {} +- {}'.format(round(x0[0]*1e3,3), round(x0[1]*1e3,3)))
        print(' y -> {} +- {}'.format(round(y0[0]*1e3,3), round(y0[1]*1e3,3)))
        print()
        envFits    = np.transpose(envFits)
        envx       = np.average(envFits[1])
        denvx_stat = np.std(envFits[1])
        denvx_fit  = np.sqrt(sum(envFits[2]**2))/len(envFits[2])
        envx = round(envx, 3)
        denvx_stat = round(denvx_stat, 4)
        denvx_fit = round(denvx_fit,  4)
        denvx = denvx_stat + denvx_fit
        denvx = round(denvx, 4)

        envy       = np.average(envFits[3])
        denvy_stat = np.std(envFits[3])
        denvy_fit  = np.sqrt(sum(envFits[4]**2))/len(envFits[4])
        envy = round(envy, 3)
        denvy_stat = round(denvy_stat, 4)
        denvy_fit = round(denvy_fit,  4)
        denvy = denvy_stat + denvy_fit
        denvy = round(denvy, 4)

        print('-------  Enveloppe [mm] ------')
        print(' x -> {} +- {}'.format(envx, denvx))
        print(' y -> {} +- {}'.format(envy, denvy))
        print('------------------------------')
        print()

    def handFitG3DG3G(self, gitterFile):
        """
        This function is to fit by hand troublesome Gridprofiles.
        Mostly in G3DG3G
        """
        info = self.getMessInfo(gitterFile)
        data = np.loadtxt(gitterFile, delimiter=';',skiprows=34,
                          encoding='latin-1', unpack=True)
        time = info[1]
        time = self.timeToMin(time)

        print('----------------------------')
        print('Monitor: {}'.format(info[0]))
        print('Time:    {}'.format(info[1]))
        print('----------------------------')
        print(gitterFile)
        print('---------------------------')
        #print('mu_x: {}'.format(info[2]))
        #print('mu_y: {}'.format(info[4]))
        # Position is the same for x and y
        posx = data[1]
        x    = data[3]
        y    = data[4]

        plt.figure(1)
        plt.plot(posx, x,
                 marker='.', markersize=8,
                 label='Wire readout', linestyle='')
        plt.xlabel('x Position [mm]')
        plt.ylabel('Intensity [a.U.]')

        plt.figure(2)
        plt.plot(posx, y,
                 marker='.', markersize=8,
                 label='Wire readout', linestyle='')
        plt.xlabel('y Position [mm]')
        plt.ylabel('Intensity [a.U.]')

        mux0, sigx0 = info[2], info[3]
        print(mux0, sigx0)
        muy0, sigy0 = info[4], info[5]
        print(muy0, sigy0)
        if(abs(sigx0) > 10.): sigx0 *= 0.1
        if(abs(sigy0) > 10.): sigy0 *= 0.1

        if(time < self.timeToMin('14:21:00') and
           time > self.timeToMin('14:16:45')):
            print(time)
            muy0 = -21.

        p0x = [max(x), mux0, sigx0, min(x)]
        p0y = [max(y), muy0, sigy0, min(y)]
        #plt.show()

        #print('x:')
        #print(p0x)
        #print('y:')
        #print(p0y)
        monitor = info[0]

        Gx, dGx = curve_fit(self.gaussCurve, posx, x, p0=p0x,maxfev=5000)
        mux  = round(Gx[1],2)
        dmux = round(np.sqrt(dGx[1][1]),2)
        Gy, dGy = curve_fit(self.gaussCurve, posx, y, p0=p0y,maxfev=5000)

        xfit = np.linspace(posx[0], posx[-1], 200)

        muy  = round(Gy[1],2)
        dmuy = round(np.sqrt(dGy[1][1]),2)

        print('mu_x: {}'.format(info[2]))
        print('Fit:  {} +- {}'.format(mux, dmux))
        print('mu_y: {}'.format(info[4]))
        print('Fit:  {} +- {}'.format(muy, dmuy))
        print('---------------------------')

        plt.figure(1)

        plt.plot(xfit, self.gaussCurve(xfit, *Gx),
                 linewidth=1.5,
                 linestyle='--',
                 label='Gauss fit')

        for i in posx: plt.axvline(i,color='gray',linewidth=0.6)
        plt.legend(loc=0)
        plt.tight_layout()
        if time < self.timeToMin('14:16:45'):
            plt.clf()

        plt.figure(2)
        plt.plot(xfit, self.gaussCurve(xfit, *Gy),
                 linewidth=1.5,
                 linestyle='--',
                 label='Gauss fit')

        for i in posx: plt.axvline(i,color='gray',linewidth=0.6)
        plt.tight_layout()
        plt.legend(loc=0)

        if time > self.timeToMin('14:16:45'):
            plt.show()
        else:
            plt.clf()
