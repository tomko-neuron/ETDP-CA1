# ----------------------------------------------------------------------------
# Simulation general parameters
data.dt = 0.2
lb.h.dt = data.dt
lb.h.steps_per_ms = 1.0 / lb.h.dt
data.recordDend = True
data.recordGDend = False
if data.actType == 'passive': data.recordGDend = False
if data.actType == 'aSoma': data.recordGDend = False

# ----------------------------------------------------------------------------
# cell type parameters: active or passive?
data.ACTIVE = True
data.ACTIVE_soma = True
data.ACTIVE_dend = True
data.ACTIVEhotSpot = False
if data.actType == 'aDend':
    data.ACTIVE = True
    data.ACTIVE_dend = True
    # data.ACTIVEhotSpot = True
if data.actType == 'aSoma':
    data.ACTIVE = True
    data.ACTIVE_soma = True
if data.actType == 'active':
    data.ACTIVE = True
    data.ACTIVE_dend = True
    data.ACTIVE_soma = True
    data.ACTIVEhotSpot = data.AHS

data.hotspot_branches = [5, 6, 52, 53, 56, 57, 59, 60, 82, 87, 111, 113]

#############################################
## FIRST: set stimulation parameters
data.st_onset = 0.1  # in seconds - only for minis
nseg_den = 10  # except when nIter

if data.stimType == 'LTP':
    data.LTP_STIMULI_NUM = 50
    data.LTP_INTERVAL = 333.33
    data.LTP_START = 12000
    data.PRELTP_TEST_NUM = 6
    data.PRELTP_TEST_INTERVAL = 2000
    data.PRELTP_TEST_START = 1000
    data.POSTLTP_TEST_NUM = 6
    data.POSTLTP_TEST_INTERVAL = 2000
    data.POSTLTP_TEST_START = 29000

    data.tInterval = .3
    data.TSTOP = 40000
    data.GABA = False
    nseg_den = 50
    data.nIter = 1  # 2

if data.stimType == 'nIter':
    if data.synType != 'single':
        print('error: synapse distribution must be single for nIter!')
        sys.exit(1)

    data.tInterval = .3
    data.TSTOP = 0.3
    data.GABA = False
    nseg_den = 50
    data.nIter = 1  # 2

if data.stimType == 'minis':
    data.TSTOP = 0.35
    data.GABA = True

### ICLAMP
data.ICLAMP = False
if data.stimType == 'SStim':
    data.ICLAMP = True
    data.iclampLoc = ['soma', 0.5]
    data.iclampOnset = 100  # ms
    data.iclampDur = 100  # ms
    data.iclampAmp = 0.3  # nA
    data.iRange = (np.arange(11) / 25.0)  # nA input resistance: mV / nA = MOhm
    # data.iRange = [0.14, 0.16] # nA
    data.TSTOP = 0.3

if data.stimType == 'DStim':
    data.ICLAMP = True
    data.iclampOnset = 25  # ms
    data.iclampDur = 2  # ms
    data.iclampAmp = 0.3  # nA
    # data.iclampLoc = ['dend', 0.5, 11]
    # data.iRange = ((np.arange(10)) / 50.0) + 0 # nA
    data.iclampLoc = ['dend', 0.5, 132]
    data.iRange = ((np.arange(4)) / 2.5) + 0.8  # nA
    data.TSTOP = 0.1  # s

if data.measureInputRes == True:
    if data.stimType == 'replay':
        data.iclampOnset = 100  # ms
        data.iclampDur = 100  # ms
        data.iclampAmp = 0.01  # nA

    if data.stimType == 'place':
        data.iclampOnset = 3000  # ms
        data.iclampDur = 3000  # ms
        data.iclampAmp = -0.01  # nA

#############################################
## SECOND: set synapse parameters
if data.model == 'CA1':
    data.cdends = [[8, 13, 36, 59], [5, 56, 144, 147], [20, 21, 45, 109], [38, 111, 127, 137], [6, 96, 146, 147],
                   [5, 36, 120, 137], [28, 56, 57, 137], [9, 48, 82, 92], [28, 96, 98, 100], [32, 93, 108, 109]]
    if data.ACTIVEhotSpot:
        data.cdends = [data.hotspot_branches]
else:
    data.cdends = [None, None, None, None, None, None, None, None, None, None]

if data.model == "CA1":
    data.Ensyn = 2000
    data.Insyn = 200
    if data.stimType == 'replay':
        data.Erate = 9
        data.Irate = 30
        data.Ensyn = int(2000 + Nclust * Ncell_per_clust)
        # data.Ensyn = int(2240)

    else:
        data.Erate = 1
        data.Irate = 7.5
    if data.stimType == 'minis':
        data.Ensyn = 100
        data.Insyn = 10

else:
    data.Ensyn = 1920
    data.Insyn = 192
    data.Erate = 5
    data.Irate = 40

data.Irev = -65
data.Brev = -80

if data.stimType == 'poisson':
    data.Erate = 1 * 10 * 0.2  # 1 Hz - average firing rate, instead of 2 synapse/micron we use 0.2 syn/um, release probability: 0.2
    data.Irate = 25

if data.synType == 'single':
    # synapse location
    data.Ensyn = 16

data.locDend = [11]  # 1dend: 11; 4dend: 11, 24, 70, 95
if data.model == "CA1":
    data.locDend = [data.iden]
    data.plotDend = [3, 4, 6, 7]

if data.locBias == 'proximal':
    data.locSeg = [0.16, 0.23]
if data.locBias == 'distal':
    data.locSeg = [0.83, 0.9]
if data.locBias == 'midle':
    data.locSeg = [0.4, 0.6]

## which dendrites to record from
if data.model == "L23":
    # data.locDendRec = [11, 9, 8, 0] 
    # data.xDendRec = [.5, .5, .5, .5]
    # best for recording - non-terminal apical branches, 100 um from soma, diam > 1 um
    # 38 - dend2_1212[0.5] L = 140 um, d=1.1 um, dist ~ 157 um
    # 47 - dend2_1222[0.5] L=150, d=1.1, dist ~ 157 um
    # 69 - dend3_1212222[0.5] L=47, d=1.4, dist ~ 135 um
    # 70 - dend3_12122221[0.5] L = 231um, d = 0.75 um
    # mainAll.model.dends[11].name()
    # mainAll.lb.h.topology()
    data.locDendRec = [11]
    # data.locDendRec = [data.locDend[0]]
    data.xDendRec = [.5]

if data.model == "CA1":
    # trunk dendrites: [-1, 76, 80, 88, 90, 94, 102, 104, 106, 110]
    # their lengths:   [57, 57, 21, 7,  31, 11, 33,  7,   5,   20]
    # data.locDendRec = [88, 110, 126, 132] # along the apical trunk 
    # data.locDendRec = [8, 13, 132, 57, 108] # along the apical trunk, a basal and an oblique
    # data.xDendRec = [.5, .5, .5, .5, .5]
    # data.locDendRec = [92, 98, 86, 87] 
    # data.locDendRec = [52] 
    # data.xDendRec = [.5]
    # data.locDendRec = [data.locDend[0], 55] # for NIter - the stimulated branch
    # data.locDendRec = [92, 98, 86, 87] 
    # data.locDendRec = [52] 
    # data.xDendRec = [ .5, .5]
    data.locDendRec = data.plotDend + data.locDend
    data.xDendRec = [0.5] * (len(data.plotDend) + 1)

    # data.locSpineRec = [900, 960, 1040, 1100]
    data.locSpineRec = [880, 914, 937, 944]

data.Itau1 = 0.1  # ms
data.Itau2 = 4  # ms

data.Btau1 = 1  # ms
data.Btau2 = 40  # ms

if (data.model == "L23"):
    data.Agmax = 0.5  # 0.5 nS
    data.Ngmax = 0.8  # 0.5 nS

    data.Igmax = 0.7  # 0.4 nS
    data.Bgmax = 0.33  # 0.4 nS

    data.Atau1 = 0.1  # ms
    data.Atau2 = 1  # ms

    data.Ntau1 = 2  # ms
    data.Ntau2 = 40  # ms

    data.Irev = -70
    data.Brev = -85

# Otmakhova et al., 2002: the NMDA/AMPA area under the curve is 5-8 at -20 mV
# Myme03 found that the conductance peak ratio is between 0.4-1.7 in neocortical cells
if (data.model == "CA1"):
    data.Agmax = 0.6 * data.g_factor  # 0.1 nS
    data.Ngmax = 0.8 * data.g_factor  # 0.8 nS
    if (data.actType == 'passive'): data.Ngmax = 0.6 * data.g_factor  # 0.6 nS
    data.Igmax = 0.7 * data.g_factor  # 0.7 nS
    data.Brev = -75  # -75
    data.Bgmax = 1.2 * data.g_factor  # 1.2 nS
    if (data.g_factor == 2):
        data.Igmax = 0.85 * data.g_factor  # 0.85 nS
        data.Brev = -75  # -75
        data.Bgmax = 1.45 * data.g_factor  # 1.45 nS

    # if (data.stimType == 'replay'):
    #     data.Bgmax = 1 #* data.g_factor  # 1.2 nS
    if (data.synType == 'somatic'):
        g_Efactor = 0.75
        data.Agmax = 0.6 * g_Efactor  # 0.1 nS
        data.Ngmax = 0.8 * g_Efactor  # 0.8 nS

    # data.Atau1 = 0.1 # ms
    # data.Atau2 = 15 # ms
    data.Atau1 = 0.1  # ms
    data.Atau2 = 1  # ms

    data.Ntau1 = 2  # ms
    data.Ntau2 = 50  # ms

# Spine dimesions - spines are hoc objects, so these parameters will be needed in the hoc environment
# data.sneck_diam = 0.04 # um (high resistance), approximate Grunditz et al model */
# sneck_diam = 0.15 # um (low resistance) */
data.sneck_diam = 0.077  # um  from Harnett et al, 2012
data.sneck_len = 1.58  # 1.58 um
data.shead_diam = 0.5  # um
data.shead_len = 0.5  # um

#############################################
## THIRD: set activity parameters
