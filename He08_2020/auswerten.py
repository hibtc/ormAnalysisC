from OrmAnalysis import OrbitResponse
import numpy as np
import matplotlib.pyplot as plt

def preGantry(simpleFit=False):

    from OrmAnalysis import ORMOptimizer
    modelPath = '/home/cristopher/HIT/hit_models/hht3/run.madx'
    # First session
    prePath ="/media/cristopher/INTENSO/ormMessungen16_08_2020/"
    profilePath = '/media/cristopher/INTENSO/ormMessungen16_08_2020/GitterProfile/'
    messFiles = [
        '2020-08-16_09-06-55_hht3_h1dg1g.orm_measurement.yml',
        '2020-08-16_09-11-26_hht3_h1dg2g.orm_measurement.yml',
        '2020-08-16_09-22-06_hht3_h2dg2g.orm_measurement.yml',
        '2020-08-16_09-35-13_hht3_h3dg3g.orm_measurement.yml',
        '2020-08-16_10-09-09_hht3_b3dg2g.orm_measurement.yml',
        '2020-08-16_10-36-07_hht3_b3dg3g.orm_measurement.yml',
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
        #'kL_B3QD11',
        #'kL_B3QD12',
    ]

    opt = ORMOptimizer(messFiles, modelPath, profilePath,
                       readOrm=False, plotEachM=False,
                       plotShots=False,
                       savePath='/home/cristopher/HIT/BerichtORM/bericht06_2020')
    singMask = 1e5
    err      = 1
    maxIts   = 100
    opt.showFit('fitStrengths.txt', saveMessFiles=False)
    if simpleFit:
        opt.fitErrorsSimple(plotMeas=1)
    else:
        opt.fitErrors(pList, singularMask=singMask,
                      error=err,  maxIt=maxIts)


def gantry(simpleFit=False):

    from OrmAnalysis import ORMOptimizer
    modelPath = '/home/cristopher/HIT/hit_models/hht3/run.madx'
    prePath ="GantryMessungen/"
    prePath ='/media/cristopher/INTENSO/ormMessungen16_08_2020/'
    profilePath = '/media/cristopher/INTENSO/ormMessungen16_08_2020/GitterProfile/'
    messFiles = [
        '2020-08-16_11-06-36_hht3_g3dg3g.orm_measurement.yml',
        '2020-08-16_11-37-45_hht3_g3dg5g.orm_measurement.yml',
        'hht3_t3df1.yml',
        'hht3_t3dg1g.yml',
        'hht3_t3dg2g.yml',
    ]

    for i in range(len(messFiles)): messFiles[i] = prePath + messFiles[i]
    pList =  [
        #'kL_H1QD11',
        #'kL_H1QD12',
        #'kL_H2QT11',
        #'kL_H2QT12',
        #'kL_H2QT13',
        #'kL_H3QD11',
        #'kL_H3QD12',
        #'kL_H3QD21',
        #'kL_H3QD22',
        'kL_B3QD11',
        'kL_B3QD12',
        #'gantry_angle',
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
                       savePath='/home/cristopher/HIT/BerichtORM/bericht06_2020')
    opt.showFit('fitStrengths.txt', saveMessFiles=False)
    singMask = 1e6
    err      = 1
    maxIts   = 100
    if simpleFit:
        opt.fitErrorsSimple(plotMeas=1)
    else:
        opt.fitErrors(pList, singularMask=singMask,
                      error=err,  maxIt=maxIts)

#preGantry(simpleFit=1)
gantry(simpleFit=1)
