from yaml import safe_load
import yaml
import copy

def separateMessung(theFile):

    with open(theFile) as f: data=safe_load(f)
    optics = data['records']
    print(len(optics))
    nullOpt = optics[0]
    shots = nullOpt['shots']
    print(len(shots))
    shot0 = shots[0]
    mons = list(shot0.keys())[:-1]
    print(mons)

    for moni in mons:
        monitor = copy.deepcopy(data)
        notMon = [monj for monj in mons
                       if moni != monj]

        optics = monitor['records']
        for opti in optics:
            shots = opti['shots']
            for shot in shots:
                for nMoni in notMon:
                    del shot[nMoni]

        with open(r'hht3_{}.yml'.format(moni), 'w') as file:
            documents = yaml.dump(monitor, file)

def main():
    prePath = './'
    messFile = '2020-08-16_12-21-36_hht3_t3df1.orm_measurement.yml'
    separateMessung(prePath+messFile)

main()
