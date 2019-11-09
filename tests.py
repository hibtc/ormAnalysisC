from OrmAnalysis import OrbitResponse
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Here are the measurements of the first session
    prePath1 ="../ormData/ormMessdata/2018-10-20-orm_measurements/M8-E108-F1-I9-G1/"
    messFiles1 = [
        '2018-10-21_04-23-18_hht3_h1dg1g.orm_measurement.yml',
        '2018-10-21_04-16-30_hht3_h1dg2g.orm_measurement.yml',
        '2018-10-21_04-08-39_hht3_h2dg2g.orm_measurement.yml',
        '2018-10-21_03-54-09_hht3_h3dg3g.orm_measurement.yml',
        '2018-10-21_03-38-51_hht3_b3dg2g.orm_measurement.yml',
        '2018-10-21_03-21-09_hht3_b3dg3g.orm_measurement.yml',
        '2018-10-21_02-50-02_hht3_g3dg3g.orm_measurement.yml',
        '2018-10-21_02-25-45_hht3_g3dg5g.orm_measurement.yml',
        #'2018-10-21_01-52-39_hht3_t3df1.orm_measurement.yml',
    ]

    prePath3 = '../ormAnalysis/ormMessdata/2019-05-11/ORM_Daten/'
    # Transfer Line Daten ohne Gantry for third session
    messFiles3 = [
        '2019-05-12_02-17-11_hht2_h1dg1g.orm_measurement.yml',
        '2019-05-12_02-21-21_hht2_h1dg2g.orm_measurement.yml',
        '2019-05-12_02-27-35_hht2_h2dg2g.orm_measurement.yml',
        '2019-05-12_02-39-53_hht3_h3dg3g.orm_measurement.yml',
        '2019-05-12_02-51-13_hht3_b3dg2g.orm_measurement.yml',
        '2019-05-12_03-05-19_hht3_b3dg3g.orm_measurement.yml',
        #'/home/cristopher/HIT/ormAnalysis/ormMessdata/2019-05-11/GantryDaten/nullDeg/2019-05-12_05-19-01_hht3_g3dg3g.orm_measurement.yml',
        #'/home/cristopher/HIT/ormAnalysis/ormMessdata/2019-05-11/GantryDaten/nullDeg/2019-05-12_05-00-51_hht3_g3dg5g.orm_measurement.yml'
    ]
    prePath4 = '../ormAnalysis/ormMessdata/10-06-2019/'
    messFiles4 = [
        '2019-06-10_08-31-35_hht3_h1dg1g.orm_measurement.yml',
        '2019-06-10_08-36-56_hht3_h1dg2g.orm_measurement.yml',
        '2019-06-10_08-48-04_hht3_h2dg2g.orm_measurement.yml',
        '2019-06-10_09-00-22_hht3_h3dg3g.orm_measurement.yml',
        '2019-06-10_09-20-44_hht3_b3dg2g.orm_measurement.yml',
        '2019-06-10_09-44-26_hht3_b3dg3g.orm_measurement.yml',
        '2019-06-10_10-10-58_hht3_g3dg3g.orm_measurement.yml',
        '2019-06-10_10-41-00_hht3_g3dg5g.orm_measurement.yml',
        ]

def testStability():

    prePath   = '../ormData/ormMessdata/10-06-2019/'
    messFile  = '2019-06-10_09-44-26_hht3_b3dg3g.orm_measurement.yml'
    # We have to load this model in order to get the sequence
    modelPath = '../hit_models/hht3/run.madx'
    orm = OrbitResponse(prePath+messFile, modelPath)
    #orm.visualizeData()
    pList =  ['ay_h1ms4']#,'ax_h1mb1']#,'ax_h1ms3','ay_h1ms4', 'kL_H1QD12']
    dpList = [0.]

    h = []
    dx = []
    dy = []

    monitor = orm.monitor
    kickers = orm.kickers

    for i in range(3,10):
        h_i = 10**(-i)
        print(h_i)
        h.append(i)
        fwdDorm  = orm.centralDorm(pList,dpList,dp=h_i)
        fwdDormx = fwdDorm[0]
        fwdDormy = fwdDorm[1]
        dx.append(fwdDormx[0])
        dy.append(fwdDormy[0])
    dx = np.transpose(dx)
    dy = np.transpose(dy)
    for i in range(len(dx)):
        c = dx[i]
        plt.plot(h[:-1],abs(c[1:]-c[:-1]),marker='.',label=kickers[i])
        plt.yscale('log')
    plt.ylabel(r'|$c_{x,i+1} - c_{x,i}$| (a.U.)')
    plt.xlabel(r'$n$')
    plt.title('Error of the ORM derivative with respect to {} at monitor {} \nwith central finite difference'.format(pList[0],monitor))
    plt.legend(loc=0)
    plt.show()

    for i in range(len(dx)):
        c = dx[i]
        plt.plot(h,c,marker='.',label=kickers[i])
    plt.title('Absolute value of the derivative with central finite difference')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$dc_x/dp$')
    plt.legend(loc=0)
    plt.show()

    for i in range(len(dy)):
        c = dy[i]
        plt.plot(h[:-1],abs(c[1:]-c[:-1]),marker='.',label=kickers[i])
        plt.yscale('log')
    plt.ylabel(r'|$c_{y,i+1} - c_{y,i}$| (a.U.)')
    plt.xlabel(r'$n$')
    plt.title('Error of the ORM derivative with respect to {} at monitor {} \nwith central finite difference'.format(pList[0],monitor))
    plt.legend(loc=0)
    plt.show()

    for i in range(len(dy)):
        c = dy[i]
        plt.plot(h,c,marker='.',label=kickers[i])
    plt.title('Absolute value of the derivative with central finite difference')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$dc_y/dp$')
    plt.legend(loc=0)
    plt.show()

def preGantry(simpleFit=False):

    from OrmAnalysis import ORMOptimizer
    modelPath = '../hit_models/hht3/run.madx'
    # First session
    prePath1 ="../ormData/ormMessdata/2018-10-20-orm_measurements/M8-E108-F1-I9-G1/"
    profilePath = '/home/cristopher/HIT/ormData/ormMessdata/10-06-2019/GitterProfile/ProfilePT3/'
    messFiles1 = [
        '2018-10-21_04-23-18_hht3_h1dg1g.orm_measurement.yml',
        '2018-10-21_04-16-30_hht3_h1dg2g.orm_measurement.yml',
        '2018-10-21_04-08-39_hht3_h2dg2g.orm_measurement.yml',
        '2018-10-21_03-54-09_hht3_h3dg3g.orm_measurement.yml',
        '2018-10-21_03-38-51_hht3_b3dg2g.orm_measurement.yml',
        '2018-10-21_03-21-09_hht3_b3dg3g.orm_measurement.yml',
    ]
    # Third session
    prePath3 = '../ormData/ormMessdata/2019-05-11/ORM_Daten/'
    messFiles3 = [
        '2019-05-12_02-17-11_hht2_h1dg1g.orm_measurement.yml',
        '2019-05-12_02-21-21_hht2_h1dg2g.orm_measurement.yml',
        '2019-05-12_02-27-35_hht2_h2dg2g.orm_measurement.yml',
        '2019-05-12_02-39-53_hht3_h3dg3g.orm_measurement.yml',
        '2019-05-12_02-51-13_hht3_b3dg2g.orm_measurement.yml',
        '2019-05-12_03-05-19_hht3_b3dg3g.orm_measurement.yml',
    ]
    # Fourth session
    prePath4 = '../ormData/ormMessdata/10-06-2019/'
    messFiles4 = [
        '2019-06-10_08-31-35_hht3_h1dg1g.orm_measurement.yml',
        '2019-06-10_08-36-56_hht3_h1dg2g.orm_measurement.yml',
        '2019-06-10_08-48-04_hht3_h2dg2g.orm_measurement.yml',
        '2019-06-10_09-00-22_hht3_h3dg3g.orm_measurement.yml',
        '2019-06-10_09-20-44_hht3_b3dg2g.orm_measurement.yml',
        '2019-06-10_09-44-26_hht3_b3dg3g.orm_measurement.yml',
        ]

    for i in range(len(messFiles1)): messFiles1[i] = prePath1 + messFiles1[i]
    for i in range(len(messFiles3)): messFiles3[i] = prePath3 + messFiles3[i]
    for i in range(len(messFiles4)): messFiles4[i] = prePath4 + messFiles4[i]

    pList =  [
        'kL_H1QD11',
        'kL_H1QD12',
        'kL_H2QT11',
        'kL_H2QT12',
        'kL_H2QT13',
        'kL_H3QD11',
        'kL_H3QD12',
        'kL_H3QD21',
        'kL_H3QD22',
        'kL_B3QD11',
        'kL_B3QD12',
    ]
    opt = ORMOptimizer(messFiles4, modelPath, profilePath,
                       readOrm=False, plotEachM=False,
                       plotShots=False)
                       #savePath='../ormBericht/NotSimpleResults/hht3')
    singMask = 1e5
    err      = 1e-1
    maxIts   = 200
    if simpleFit:
        opt.fitErrorsSimple(plotMeas=1)
    else:
        opt.fitErrors(pList, singularMask=singMask,
                      error=err,  maxIt=maxIts)

def gantry(simpleFit=False):
    
    from OrmAnalysis import ORMOptimizer
    modelPath = 'hht3/run.madx'
    profilePath = '/home/cristopher/HIT/ormData/ormMessdata/10-06-2019/GitterProfile/ProfilePT3/'
    # First session
    prePath = '../ormData/ormMessdata/10-06-2019/'
    messFiles = [
        '2019-06-10_08-31-35_hht3_h1dg1g.orm_measurement.yml',
        '2019-06-10_08-36-56_hht3_h1dg2g.orm_measurement.yml',
        '2019-06-10_08-48-04_hht3_h2dg2g.orm_measurement.yml',
        '2019-06-10_09-00-22_hht3_h3dg3g.orm_measurement.yml',
        '2019-06-10_09-20-44_hht3_b3dg2g.orm_measurement.yml',
        '2019-06-10_09-44-26_hht3_b3dg3g.orm_measurement.yml',
        '2019-06-10_10-10-58_hht3_g3dg3g.orm_measurement.yml',
        '2019-06-10_10-41-00_hht3_g3dg5g.orm_measurement.yml',
        '2019-06-10_11-13-49_hht3_t3dg2g.orm_measurement.yml',
        '2019-06-10_11-33-56_hht3_t3df1.orm_measurement.yml',
        ]
    for i in range(len(messFiles)): messFiles[i] = prePath + messFiles[i]
    pList =  [
        'kL_H1QD11',
        'kL_H1QD12',
        'kL_H2QT11',
        'kL_H2QT12',
        'kL_H2QT13',
        'kL_H3QD11',
        'kL_H3QD12',
        'kL_H3QD21',
        'kL_H3QD22',
        'kL_B3QD11',
        'kL_B3QD12',
        'gantry_angle',
        'kL_G3QD11',
        'kL_G3QD12',
        'kL_G3QD21',
        'kL_G3QD22',
        'kL_G3QD31',
        'kL_G3QD32',
        'kL_EFG_G3QD41',
        'kL_EFG_G3QD42',
    ]
    opt = ORMOptimizer(messFiles, modelPath, profilePath,
                       readOrm=False, plotEachM=False,
                       savePath='../ormBericht/NotSimpleResults/gantry')
    singMask = 1e6
    err      = 1.
    maxIts   = 200
    if simpleFit:
        opt.fitErrorsSimple(plotMeas=1)
    else:
        opt.fitErrors(pList, singularMask=singMask,
                      error=err,  maxIt=maxIts)
    
def fit_hht2(simpleFit=False):
    
    from OrmAnalysis import ORMOptimizer, OrbitResponse
    prePath = '../ormData/ormMessdata/10-06-2019/'
    modelPath_hht2 = '../hit_models/hht2/run.madx'
    gitterPath = '/home/cristopher/HIT/ormData/ormMessdata/10-06-2019/GitterProfile/ProfilePT2/'
    hht2 = [
        '2019-06-10_13-35-08_hht2_h1dg1g.orm_measurement.yml',
        '2019-06-10_13-37-35_hht2_h1dg2g.orm_measurement.yml',
        '2019-06-10_13-46-05_hht2_h2dg2g.orm_measurement.yml',
        '2019-06-10_13-51-36_hht2_b2dg2g.orm_measurement.yml',
        '2019-06-10_13-59-38_hht2_b2dg3g.orm_measurement.yml',
        '2019-06-10_14-08-29_hht2_t2dg2g.orm_measurement.yml',
        '2019-06-10_14-18-50_hht2_t2df1.orm_measurement.yml',
    ]
    
    for i in range(len(hht2)): hht2[i] = prePath + hht2[i]
    
    pList_hht2 =  [
        'kL_H1QD11',
        'kL_H1QD12',
        'kL_H2QT11',
        'kL_H2QT12',
        'kL_H2QT13',
        'kL_B2QD11',
        'kL_B2QD12',
        'kL_B2QD21',
        'kL_B2QD22',
    ]
    opt = ORMOptimizer(hht2, modelPath_hht2, gitterPath,
                       readOrm=False, plotEachM=False)
    singMask = 1e6
    err      = 1e-1
    maxIts   = 200
    if simpleFit:
        opt.fitErrorsSimple(plotMeas=True)
    else:
        opt.fitErrors(pList_hht2, singularMask=singMask,
                      error=err,  maxIt=maxIts)

def fit_hht1(simpleFit=False):
    from OrmAnalysis import ORMOptimizer, OrbitResponse
    prePath = '../ormData/ormMessdata/10-06-2019/'
    modelPath_hht1 = '../hit_models/hht1/run.madx'
    gitterPath = '/home/cristopher/HIT/ormData/ormMessdata/10-06-2019/GitterProfile/ProfilePT1/'
    hht1 = [
        '2019-06-10_12-08-47_hht1_h1dg1g.orm_measurement.yml',
        '2019-06-10_12-12-52_hht1_h1dg2g.orm_measurement.yml',
        '2019-06-10_12-27-49_hht1_b1dg2g.orm_measurement.yml',
        '2019-06-10_12-44-54_hht1_b1dg3g.orm_measurement.yml',
        '2019-06-10_13-04-07_hht1_t1dg2g.orm_measurement.yml',
        '2019-06-10_13-17-21_hht1_t1df1.orm_measurement.yml',
    ]
    for i in range(len(hht1)): hht1[i] = prePath + hht1[i]
    pList_hht1 =  [
         'kL_H1QD11',
         'kL_H1QD12',
         'kL_B1QD11',
         'kL_B1QD12',
         'kL_B1QD21',
         'kL_B1QD22',
     ]
    opt = ORMOptimizer(hht1, modelPath_hht1, gitterPath,
                       readOrm=False, plotEachM=False)
    singMask = 1e5
    err      = 1e-1
    maxIts   = 200
    if simpleFit:
        opt.fitErrorsSimple(plotMeas=True)
    else:
        opt.fitErrors(pList_hht1, singularMask=singMask,
                      error=err,  maxIt=maxIts)

def testFit():

    from OrmAnalysis import ORMOptimizer
    modelPath = '../hit_models/hht3/run.madx'
    profilePath = '/home/cristopher/HIT/ormData/ormMessdata/10-06-2019/GitterProfile/ProfilePT3/'
    # Last session
    prePath1 ="../ormData/ormMessdata/10-06-2019/"
    messFiles1 = [
        '2019-06-10_08-31-35_hht3_h1dg1g.orm_measurement.yml',
        '2019-06-10_08-36-56_hht3_h1dg2g.orm_measurement.yml',
        #'2019-06-10_08-48-04_hht3_h2dg2g.orm_measurement.yml',
        #'2019-06-10_09-00-22_hht3_h3dg3g.orm_measurement.yml',
        #'2019-06-10_09-20-44_hht3_b3dg2g.orm_measurement.yml',
        #'2019-06-10_09-44-26_hht3_b3dg3g.orm_measurement.yml',
        #'2019-06-10_10-10-58_hht3_g3dg3g.orm_measurement.yml',
        #'2019-06-10_10-41-00_hht3_g3dg5g.orm_measurement.yml',
        '2019-06-10_11-13-49_hht3_t3dg2g.orm_measurement.yml',
        '2019-06-10_11-33-56_hht3_t3df1.orm_measurement.yml',
        ]
    for i in range(len(messFiles1)): messFiles1[i] = prePath1 + messFiles1[i]
    pList =  ['kL_H1QD11', 'kL_H1QD12']
    pList =  [
        'kL_H1QD11',
        'kL_H1QD12',
        #'kL_H2QT11',
        #'kL_H2QT12',
        #'kL_H2QT13',
        #'kL_H3QD11',
        #'kL_H3QD12',
        #'kL_H3QD21',
        #'kL_H3QD22',
        #'kL_B3QD11',
        #'kL_B3QD12',
        #'kL_G3QD11',
        #'kL_G3QD12',
        #'kL_G3QD21',
        #'kL_G3QD22',
        #'kL_G3QD31',
        #'kL_G3QD32',
        #'kL_EFG_G3QD41',
        #'kL_EFG_G3QD42',
    ]
    opt = ORMOptimizer(messFiles1, modelPath, profilePath,
                       readOrm=False, plotEachM=False,
                       plotShots=False, savePath='.')
    #opt.fitErrorsSimple()
    singMask = 1e6
    err      = 1e1
    maxIts   = 50
    opt.fitErrors(pList, singularMask=singMask,
                  error=err,  maxIt=maxIts,
                  continueFit=True)

def testGitterReader():
    from OrmAnalysis import ORMOptimizer, OrbitResponse
    from ProfileAnalyzer import ProfileAnalyzer    
    gitterPath = '/home/cristopher/HIT/ormData/ormMessdata/10-06-2019/GitterProfile/'
    gitterProfile = 'ProfilePT2/'

    prePath = '../ormData/ormMessdata/10-06-2019/'
    modelPath_hht1 = '../hit_models/hht1/run.madx'
    hht1 = [
        '2019-06-10_12-08-47_hht1_h1dg1g.orm_measurement.yml',
        '2019-06-10_12-12-52_hht1_h1dg2g.orm_measurement.yml',
        '2019-06-10_12-27-49_hht1_b1dg2g.orm_measurement.yml',
        '2019-06-10_12-44-54_hht1_b1dg3g.orm_measurement.yml',
    ]
    modelPath_hht2 = '../hit_models/hht2/run.madx'
    hht2 = [
        #'2019-06-10_13-35-08_hht2_h1dg1g.orm_measurement.yml',
        #'2019-06-10_13-37-35_hht2_h1dg2g.orm_measurement.yml',
        '2019-06-10_13-46-05_hht2_h2dg2g.orm_measurement.yml',
        '2019-06-10_13-51-36_hht2_b2dg2g.orm_measurement.yml',
        '2019-06-10_13-59-38_hht2_b2dg3g.orm_measurement.yml',
    ]
    modelPath_hht3 = '../hit_models/hht3/run.madx'
    hht3 = [
        '2019-06-10_08-31-35_hht3_h1dg1g.orm_measurement.yml',
        '2019-06-10_08-36-56_hht3_h1dg2g.orm_measurement.yml',
        '2019-06-10_08-48-04_hht3_h2dg2g.orm_measurement.yml',
        '2019-06-10_09-00-22_hht3_h3dg3g.orm_measurement.yml',
        '2019-06-10_09-20-44_hht3_b3dg2g.orm_measurement.yml',
        '2019-06-10_09-44-26_hht3_b3dg3g.orm_measurement.yml',
        '2019-06-10_10-10-58_hht3_g3dg3g.orm_measurement.yml',
        '2019-06-10_10-41-00_hht3_g3dg5g.orm_measurement.yml',
        ]
    for mFile in hht3:
        madguiData = prePath+mFile
        ormA = OrbitResponse(madguiData, modelPath_hht3, gitterPath)
        data = ormA.data
        gR = ProfileAnalyzer(data, gitterPath+gitterProfile)
        gR.fitProfiles(ormA.monitor.upper(), showProfiles=0,
                       skipShots=1)

#testGitterReader()
#fit_hht1(simpleFit=0)
#fit_hht2(simpleFit=0)
#preGantry(simpleFit=0)
gantry(simpleFit=0)
#testFit()
