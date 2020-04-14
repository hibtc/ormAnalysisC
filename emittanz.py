from madgui.core.app import init_app
from madgui.core.session import Session
import madgui.util.yaml as yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from scipy.stats import moyal

modelPath = '/home/cristopher/HIT/hit_models'

def app(): return init_app([], gui=False)

class EmittanzRechner:

    def __init__(self, app):
        self.session = Session()
        self.loadModel()
        self.model   = self.session.model()
        self.globs   = self.model.globals

    def loadStrengths(self, filename): self.model.load_strengths(filename)
    def loadModel(self): self.session.load_model(modelPath, stdout=False)

    def getSecMap(self, initElem, endElem):
        return self.model.sectormap(initElem, endElem)

    def _buildMats(self, secMaps):
        Mx = [[m[0][0]**2,
               m[0][1]*m[0][0]*2,
               m[0][1]**2]
              for m in secMaps
              ]

        My = [[m[2][2]**2,
               m[2][3]*m[2][2]*2,
               m[2][3]**2]
                for m in secMaps]
        return Mx, My

    def computeTwiss(self, sig11, sig12, sig22):
        em2 = sig11*sig22 - sig12**2
        if em2 < 0.:
            return  [0.]*3
        beta  = sig11/np.sqrt(em2)
        alpha = -sig12/np.sqrt(em2)
        gamma = sig22/np.sqrt(em2)
        return [beta, alpha, gamma]

    def computeEmit(self, sig11, sig12, sig22):
        em2 = sig11*sig22 - sig12**2
        if em2 < 0.:
            return 0.
        return np.sqrt(em2)

    def _solveLinEqs(self, M, x):
        Mt = np.transpose(M)
        M  = np.matmul(np.linalg.inv(np.matmul(Mt,M)), Mt)
        return np.matmul(M,x)

    def threeMonsMethod(self, data):
        """
        @param array data: should have the following structure
               [[monitor1, [sigx1 , sigy1]],
                [monitor2, [sigx2 , sigy2]],
                [monitor3, [sigx3 , sigy3]]]
        """
        mes = np.array(data)
        mes = np.transpose(mes)
        mons = mes[0]
        beam = mes[1]

        secMaps = [self.getSecMap('hht3$start', mi) for mi in mons]

        Mx, My = self._buildMats(secMaps)
        x,  y  = [b[0] for b in beam], [b[1] for b in beam]

        solx = self._solveLinEqs(Mx,x)
        soly = self._solveLinEqs(My,y)

        return self.computeEmit(*solx), self.computeEmit(*soly)

    def sample(self, mu, sig, N=1000):
        return np.random.normal(mu, sig, N)

    def gauss(self, x, mu, sig, A):
        return A*np.exp(-(x-mu)**2/(2*sig**2))

    def moyal(self, x, mu, sig, A):
        return A*np.exp(-( (x-mu)/sig + np.exp(-(x-mu)/sig) )/2)

    def _showMonteCarl(self, emx, emy):
        print('RESULTS:')
        print('emx: {} +- {}'.format(*emx))
        print('emy: {} +- {}'.format(*emy))

    def _plotTwissx(self, twiss):
        twiss = np.transpose(twiss)
        plt.figure('Beta_x')

        beta = twiss[0]
        mask = beta < 200.
        beta_masked = beta[mask]
        b = plt.hist(beta_masked, bins=100)
        x = (b[1][:-1] + b[1][1:])/2

        mu = np.mean(beta)
        sig = np.std(beta)
        A   = max(b[0])

        popt, pcov = curve_fit(self.moyal, x, b[0], p0=[mu, sig, A])
        print('Beta:')
        print()
        print(popt)
        print()

        x = np.linspace(0, 100, 200)

        plt.plot(x, self.moyal(x, *popt))


        plt.figure('Alpha_x')

        alf = twiss[1]
        alf = alf[twiss[0] < 200.]
        a   = plt.hist(alf, bins=100)

        popt = self.fitData(alf, moyalFit=True)
        x    = -a[1]
        plt.plot(-x, self.moyal(x, *popt))
        print('Alpha')
        print()
        print(popt)
        plt.show()

    def _plotTwissy(self, twiss):

        bety_max = 15.
        twiss = np.transpose(twiss)
        plt.figure('Beta_x')

        beta = twiss[0]
        mask = beta < bety_max
        beta_masked = beta[mask]
        b = plt.hist(beta_masked, bins=100)
        x = (b[1][:-1] + b[1][1:])/2

        mu = np.mean(beta)
        sig = np.std(beta)
        A   = max(b[0])

        popt, pcov = curve_fit(self.moyal, x, b[0], p0=[mu, sig, A])
        print('Beta_y:')
        print()
        print(popt)
        print()

        x = np.linspace(0, bety_max, 200)

        plt.plot(x, self.moyal(x, *popt))


        plt.figure('Alpha_y')

        alfy_max = 1.
        alf = twiss[1]
        alf = alf[twiss[1] < alfy_max]
        a   = plt.hist(alf, bins=100)
        x   = (a[1][:-1] + a[1][1:])/2

        mu  = np.mean(alf)
        sig = np.std(alf)
        A   = max(a[0])

        popt, pcov = curve_fit(self.moyal, x, a[0], p0=[mu, sig, A])

        x = np.linspace(-1., 3, 200)
        plt.plot(x, self.moyal(x, *popt))
        print('Alpha')
        print()
        print(popt)
        plt.show()

    def fitData(self, g, moyalFit=False):
        h = plt.hist(g, bins=100)
        x = (h[1][:-1] + h[1][1:])/2
        y = h[0]

        if moyalFit:
            x  *= -1
            mu = np.mean(x)
            sig = np.std(x)
            A   = max(y)

            p = [mu, sig, A]
            popt, pcov = curve_fit(self.moyal, x, y, p0=p)
        else:
            mu = np.mean(x)
            sig = np.std(x)
            A   = max(y)
            p = [mu, sig, A]
            popt, pcov = curve_fit(self.gauss, x, y, p0=p)

        return popt

    def monteCarl(self, data, unc, plot=False):
        mes = np.array(data)
        mes = np.transpose(mes)
        mons = mes[0]
        beam = mes[1]
        N = 100000

        xSample = [self.sample(beam[i][0], unc[i][0], N)
                   for i in range(len(beam))]
        ySample = [self.sample(beam[i][1], unc[i][1], N)
                   for i in range(len(beam))]

        xSample = np.transpose(xSample)
        ySample = np.transpose(ySample)

        secMaps = [self.getSecMap('hht3$start', mi) for mi in mons]
        Mx, My = self._buildMats(secMaps)

        solx = [self._solveLinEqs(Mx, x) for x in xSample]
        emx  = np.array([self.computeEmit(*sx) for sx in solx])
        twx  = np.array([self.computeTwiss(*sx) for sx in solx])
        soly = [self._solveLinEqs(My, y) for y in ySample]
        emy  = np.array([self.computeEmit(*sy) for sy in soly])
        twy  = np.array([self.computeTwiss(*sy) for sy in soly])

        twx = twx[emx > 0.]
        twy = twy[emy > 0.]
        emx = emx[emx > 0.]
        emy = emy[emy > 0.]

        fitx = self.fitData(emx*1e6)
        fity = self.fitData(emy*1e6, moyalFit=True)
        self._showMonteCarl(fitx[:2], fity[:2])
        self._plotTwissx(twx)
        self._plotTwissy(twy)

        if plot:
            plt.cla()
            plt.figure(1)
            plt.xlabel(r'$\varepsilon_x$ [mm mrad]')
            h = plt.hist(emx*1e6, bins=100, alpha=0.6)
            x = np.linspace(0., max(h[1]), 200)
            plt.plot(x, self.gauss(x, *fitx))
            print('Emx')
            print(fitx)

            plt.figure(2)
            plt.xlabel(r'$\varepsilon_y$ [mm mrad]')
            h = plt.hist(emy*1e6, bins=100, alpha=0.6)
            x = np.linspace(0., max(h[1]), 200)
            x *= -1
            plt.plot(-x, self.moyal(x, *fity))
            print('Emy')
            print(fity)
            plt.show()

def intensity8():

    homy = "/home/cristopher/HIT/emittanzMess/OrbitResponse/"
    messStrengths = homy + "strengths.yml"
    fitStrengths  = homy + "fit_I8.yml"
    emRechner = EmittanzRechner(app())
    emRechner.loadStrengths(messStrengths)
    emRechner.loadStrengths(fitStrengths)

    # These are the measurements
    mons = ['h1dg1g', 'h1dg2g', 'h2dg2g', 'h3dg3g',
            'g3dg3g', 'g3dg5g']
    sigx = np.array([2.118e-3, 2.641e-3, 3.97e-3,
                     7.54e-3,  1.32e-3,  5.8e-3])**2

    sigy = np.array([2.42e-3, 3.975e-3, 6.775e-3,
                     2.25e-3, 3.36e-3,  2.09e-3])**2

    dsig = [[0.13e-3, 0.15e-3],
            [0.17e-3, 0.24e-3],
            [0.16e-3, 0.33e-3],
            [0.34e-3, 0.34e-3],
            [0.15e-3, 0.87e-3],
            [0.25e-3, 0.08e-3],
    ]

    for i in range(6):
        dsig[i][0] *= 2*3*np.sqrt(sigx[i])
        dsig[i][1] *= 2*3*np.sqrt(sigy[i])

    measurements = [
        [mons[i], [sigx[i], sigy[i]]] for i in range(6)
    ]

    for i in range(3, 7):
        print()
        print('Fitting with {} monitors'.format(i))
        print()
        m = emRechner.threeMonsMethod(measurements[:i])
        print('emx = {} mm mrad'.format(round(m[0]*1e6,4)))
        print('emy = {} mm mrad'.format(round(m[1]*1e6,4)))
        emRechner.monteCarl(measurements[:i], dsig[:i], plot=True)

def intensity9():

    homy = "/home/cristopher/HIT/emittanzMess/OrbitResponse/"
    messStrengths = homy + "strengths.yml"
    fitStrengths  = homy + "fit2_I9.yml"
    emRechner = EmittanzRechner(app())
    emRechner.loadStrengths(messStrengths)
    emRechner.loadStrengths(fitStrengths)
    mons = ['h1dg1g', 'h1dg2g', 'h2dg2g', 'h3dg3g',
            'g3dg3g', 'g3dg5g']
    sigx = np.array([2.06e-3, 2.71e-3, 3.88e-3,
                     7.898e-3,  4.05e-3,  6.396e-3])**2

    sigy = np.array([2.688e-3, 4.63e-3, 9.17e-3,
                     1.869e-3, 6.67e-3, 2.45e-3])**2

    dsig = [[0.11e-3,  0.086e-3],
            [0.19e-3,  0.18e-3],
            [0.14e-3,  0.2e-3],
            [0.836e-3, 1.46e-3],
            [0.96e-3,  1.45e-3],
            [0.49e-3,  0.71e-3],
    ]

    for i in range(6):
        dsig[i][0] *= 2*3*np.sqrt(sigx[i])
        dsig[i][1] *= 2*3*np.sqrt(sigy[i])

    measurements = [
        [mons[i], [sigx[i], sigy[i]]] for i in range(6)
    ]

    for i in range(3, 7):
        print()
        print('Fitting with {} monitors'.format(i))
        print()
        m = emRechner.threeMonsMethod(measurements[:i])
        print('emx = {} mm mrad'.format(round(m[0]*1e6,4)))
        print('emy = {} mm mrad'.format(round(m[1]*1e6,4)))
        emRechner.monteCarl(measurements[:i], dsig[:i], plot=True)

intensity8()
