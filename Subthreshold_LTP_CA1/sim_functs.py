#----------------------------------------------------------------------------
# Functions and Classes
import struct
import sys
from numpy import loadtxt
import numpy as np
import copy

def initOnsetSpikes():
    model.ncAMPAlist[0].event(data.st_onset * 1000)

def initSpikes():
    if (len(data.etimes)>0):
        for s in data.etimes:
            model.ncAMPAlist[int(s[0])].event(float(s[1]))
            if data.NMDA: model.ncNMDAlist[int(s[0])].event(float(s[1]))
    if data.GABA == True:
        if (len(data.itimes)>0):
            # print data.itimes[1]
            for s in data.itimes:
                model.ncGABAlist[int(s[0])].event(float(s[1]))
                if data.GABA_B: model.ncGABA_Blist[int(s[0])].event(float(s[1]))

def storeSimOutput(v,vD):
        data.vdata.append(v)
        data.vDdata.append(vD)

def storeSimInputOutput(v,vD,Et,It):
        data.vdata.append(v)
        data.vDdata.append(vD)
        if (len(data.stim) == 0):
            nrep = 1
        else :
            nrep = max(data.stim[:,0]) + 1
            
        if not Et is None:
            Et = np.column_stack((nrep * np.ones(len(Et[:,0])), Et))
            if (len(data.stim) == 0):
                data.stim = Et
            else :
                data.stim = np.row_stack((data.stim, Et))
        if not It is None :
            It = np.column_stack((-1 * nrep * np.ones(len(It[:,0])), It))
            if (len(data.stim) == 0):
                data.stim = It
            else :
                data.stim = np.row_stack((data.stim, It))

def storeSimInputOutputGdend(v,vD,Et,It,Gdend):
        data.vdata.append(v)
        data.vDdata.append(vD)
        data.Gdata.append(Gdend)
        if (len(data.stim) == 0):
            nrep = 1
        else :
            nrep = max(data.stim[:,0]) + 1
            
        if not Et is None:
            Et = np.column_stack((nrep * np.ones(len(Et[:,0])), Et))
            if (len(data.stim) == 0):
                data.stim = Et
            else :
                data.stim = np.row_stack((data.stim, Et))
        if not It is None :
            It = np.column_stack((-1 * nrep * np.ones(len(It[:,0])), It))
            if (len(data.stim) == 0):
                data.stim = It
            else :
                data.stim = np.row_stack((data.stim, It))

def storeSimInputOutputGdendIdend(v,vD,Et,It,Gdend, Idend):
        data.vdata.append(v)
        data.vDdata.append(vD)
        data.Gdata.append(Gdend)
        data.Idata.append(Idend)
        if (len(data.stim) == 0):
            nrep = 1
        else :
            nrep = max(data.stim[:,0]) + 1
            
        if not Et is None:
            Et = np.column_stack((nrep * np.ones(len(Et[:,0])), Et))
            if (len(data.stim) == 0):
                data.stim = Et
            else :
                data.stim = np.row_stack((data.stim, Et))
        if not It is None :
            It = np.column_stack((-1 * nrep * np.ones(len(It[:,0])), It))
            if (len(data.stim) == 0):
                data.stim = It
            else :
                data.stim = np.row_stack((data.stim, It))

#-----------------------------------------------
# Synapse location functions
#-----------------------------------------------

def genDendLocs(dends, nsyn, spread):
    # insert nsyn synapses to dendrites dends, uniform spread within a branch
    locs = []
    n_dends = len(dends)
    if isinstance(nsyn, list):
        nsyn = np.repeat(nsyn[0], len(dends))
    else:
        nsyn = np.repeat(nsyn, len(dends))        
    for i_dend in np.arange(0, n_dends):
        dend = dends[i_dend]
        nsyn_dend = nsyn[i_dend]
        isd = (spread[1]-spread[0])/float(nsyn_dend)
        pos = np.arange(spread[0], spread[1], isd)[0:nsyn_dend] 

        if len(pos) != nsyn_dend:
            print('error: synapse number mismatch, stop simulation! dend:', i_dend, 'created=', len(pos), '!=', nsyn_dend)
            sys.exit(1)
        for p in pos:
            locs.append([dend, p])

    return locs



def genDendSomaLocs(dends, nsyn, spread):
    # insert nsyn[0] synapse at the soma and nsyn synapses to dendrites dends, uniform spread
    locs = []
    n_dends = len(dends)
    nsyn_soma = nsyn[0]
    nsyn_dends = np.delete(nsyn, 0)
    for p in np.arange(0,nsyn_soma):
        locs.append([-1, 0.5])# -1 is not an index - it means that the synapse is at the soma
    for i_dend in np.arange(0,n_dends):
        dend = dends[i_dend]
        nsyn_dend = nsyn_dends[i_dend]
        isd = (spread[1]-spread[0])/(nsyn_dend)
        pos = np.arange(spread[0], spread[1], isd)[0:nsyn_dend] 

        if (len(pos) != nsyn_dend):
            print('error: synapse number mismatch, stop simulation! dend:', i_dend, 'created=', len(pos), '!=', nsyn_dend)
            sys.exit(1)
        for p in pos:
            locs.append([dend, p])
    return locs

def genSomaLocs(nsyn):
    ## insert nsyn synapse at the soma
    locs = []
    if len(nsyn) > 1:
        nsyn = sum(nsyn)
    for p in np.arange(0,nsyn):
        locs.append([-1, 0.5])# -1 is not an index - it means that the synapse is at the soma
    return locs


def genRandomLocs(nsyn, seed, dend_ids=None):
    # randomly choose a dendritic branch, proportionally to its length
    np.random.seed(seed)
    nden = len(model.dends)
    Ldends = np.zeros(nden)
    for ii in np.arange(0, nden): Ldends[ii] = model.dends[ii].L

    # bfname='Dend_length.bin'
    # binfile = file(bfname, 'wb')
    # # and write out two integers with the row and column dimension
    # header = struct.pack('2I', Ldends.shape[0], 1)
    # binfile.write(header)
    # ddata = struct.pack('%id' % Ldends.shape[0], *Ldends)
    # binfile.write(ddata)
    # binfile.close()

    # print (dend_ids)
    Pdends = Ldends / sum(Ldends)
    if dend_ids is not None:
        Pdends = np.zeros(nden)
        for dend in dend_ids:
            Pdends[dend] = Ldends[dend] / sum(Ldends)
        Pdends = Pdends / sum(Pdends)

    # print(Pdends)    
    Ndends = np.random.multinomial(nsyn, Pdends)

    #  and insert a nsyn synapses at random locations within the branch
    locs = []
    for dend in np.arange(0, nden): # 
        nsyn_dend = Ndends[dend]
        if (nsyn_dend > 0):
            locs_dend = np.sort(np.random.uniform(0, 1, nsyn_dend))
            for s in np.arange(0,nsyn_dend):
                locs.append([dend, locs_dend[s]])
    # print locs
    return locs

def genClusts(Nclust, Ncell_per_clust, minL, seed, clocs=None):
    # randomly choose a dendritic branch, larger than minL for clustered synapses
    # clocs: 
    if (minL < Ncell_per_clust):
        print('error: cluster size mismatch, stop simulation! Number of cells per cluster (1 um inter-spine distance):', Ncell_per_clust, ', minL:', minL)
        sys.exit(1)
    np.random.seed(seed)
    nden = len(model.dends)
    Ldends = np.zeros(nden)
    for ii in np.arange(0, nden): Ldends[ii] = model.dends[ii].L
    # idends = np.flatnonzero((Ldends < minL) & (Ldends > 10)) # to put clusters on short segments
    idends = np.flatnonzero(Ldends > minL)
    # print(idends)

    replace_clusts = False
    if (Nclust > len(idends)):
        print('warning: cluster number mismatch, multiple clusters are allocated to the same branch! Nclust:', Nclust, ', number of available branches:', len(idends), ', minL:', minL)
        replace_clusts = True
    
    # if some dendrites are preselected ...
    if clocs is not None:
        k = len(clocs)
        clDends1 = clocs # we choose those dendrites
        if (Nclust > k) : # we choose randomly from the others
            if (replace_clusts == False):
                idends = np.setdiff1d(idends, clocs)
            clDends2 = np.random.choice(idends, Nclust-k, replace=replace_clusts)
            clDends = np.concatenate((clDends1, clDends2))
        else:
            clDends = np.array(clDends1)

    else:
        clDends = np.random.choice(idends, Nclust, replace=replace_clusts)
    #  and insert a nsyn synapses at random locations within the branch
    # print(clDends)
    locs = []
    i_dend = 0
    for dend in clDends: # 
        locstart = np.random.uniform(0, 1 - Ncell_per_clust / Ldends[dend], 1)
        if clocs is not None:
            if i_dend < k:
                locstart = (1 - Ncell_per_clust / Ldends[dend])/2
        for s in np.arange(0,Ncell_per_clust):
            locs.append([dend, np.asscalar(s/Ldends[dend] + locstart)]) # 1 um distance between spines
        i_dend = i_dend + 1
    return locs

##########################
## these functions are needed to distribute cells assemblies uniformly within a branch
## for L synapses on a branch first select a number P that is
##      - co-prime with L
##      - somehwat smaller than L/4
## the idea is to construct a {sequence with increments of P} %% L
## and use it to index the contacts
## we only do this if L > 6

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def coprime(a, b):
    return gcd(a, b) == 1


def find_coprime(a, primerange=[3, 11]):
    if (a==8):
        return 3
    if (a==12):
        return 5
    denominators = np.arange(primerange[0], primerange[1]+1)
    nprime = len(denominators)
    coprimes = np.zeros(nprime)
    for ii in range(nprime):
        if coprime(a, denominators[ii]):
            coprimes[ii] = 1
    prime_denoms = denominators[np.where(coprimes)]
    prime_dist4 = np.square((prime_denoms - a/4.0))

    if ((a > 20) & (min(prime_dist4) < 1./15)):
        ii = np.where(prime_dist4 < 1./15)[0][0]
        prime_dist4[ii] = max(prime_dist4)
    prime = prime_denoms[np.argmin(prime_dist4)]
    return prime



def genLocalRegularLocs(nsyn, seed, Ngroups=40, regular_location=False):
    ## start from a random connectivity
    Ncell_per_group = nsyn / Ngroups
    locs = genRandomLocs(nsyn, seed=seed)
    np.random.shuffle(locs)
    newlocs = list(locs)

    ## find inputs to each dendrites
    el = np.asarray(locs)
    idend_all = el[:,0]
    locsyn_all = el[:,1]

    dends_with_syn = np.unique(idend_all)

    ## for all branches, distribute inputs uniformly - checked, it actually works
    for idend in dends_with_syn:
        indsyn_d = np.where(idend_all == idend)[0]
        L = len(indsyn_d)
    
        if (L>6):
            minL = min(1, np.floor_divide(L, 7))
            maxL = max(np.floor_divide(L, 2), 10)
            prime = find_coprime(L, [minL, maxL])

            startind = np.random.randint(L)
            regular_inds = np.mod(np.arange(start=startind, stop=startind + L*prime, step=prime), L)

            if (regular_location == True):
                loc_sorted = np.true_divide((np.arange(L)+1), np.real(L+1))
            else :
                locsyn_d = locsyn_all[indsyn_d]
                loc_sorted = np.sort(locsyn_d)

            for isyn in range(L):
                kk = np.where(regular_inds == isyn)[0][0]
                # kk2 = regular_inds[isyn]
                # print(kk, kk2)
                newlocs[indsyn_d[isyn]] = [idend, loc_sorted[kk]]
    return newlocs

def genUnifLocs(isd=1, Nsyn=2000, Ngroups=40, prime=9, locSeg=[0,1], localRandom=False):   # intersynaptic distance in microns 
    # uniform synapse density on all branches - 
    # print 'locSeg: ', locSeg[0], locSeg[1]
    
    ## first we select synapse locations on dendritic branches 
    ## sequentially with identical inter-synapse distance
    ##      so dendrites 0-1-2... will receive inputs sequentailly from assembly 1-2-3...
    locs = []
    dend_n = 0
    for dend in model.dends:
        distance = locSeg[0]*dend.L + isd
        while distance<(dend.L * locSeg[1]):
            locs.append([dend_n, distance/dend.L])
            distance = distance + isd
        dend_n = dend_n + 1

    if (len(locs) != Nsyn):
        print('error: length of locs is not the same as Nsyn', len(locs), Nsyn)
        sys.exit(1)
    
    ## second, we take groups of 40 synapses (matching the 40 assemblies)
    ##  - form sequences without clusters  
    ##  - place these sequences on contiguous dendritic segments
    newlocs = list(locs)
    randshift = np.random.randint(0, Nsyn)
    classes = np.mod(np.arange(start=1, stop=Ngroups*prime, step=prime), Ngroups)
    k = 0  
    # locsyns = []  
    for irep in np.arange(Nsyn / Ngroups):
        indlocs = irep + classes * Nsyn / Ngroups
        for isyn in indlocs:
            locsyn = np.mod(isyn + randshift, Nsyn)
            newlocs[locsyn] = locs[k]
            # newlocs.append(locs[locsyn])
            # locsyns.append(locsyn)
            k = k + 1

    ## we randomize synapses locally within each branch
    if (localRandom == True):
        el = np.asarray(newlocs)
        idend_all = el[:,0]
        locsyn_all = el[:,1]

        dends_with_syn = np.unique(idend_all)

        for idend in dends_with_syn:
            indsyn_d = np.where(idend_all == idend)[0]
            L = len(indsyn_d)
            locsyn_d = np.random.uniform(size=L)
            for isyn in range(L):
                newlocs[indsyn_d[isyn]] = [idend, locsyn_d[isyn]]

    return newlocs # list of [dend distance] pairs

# for i in np.arange(2000):
#     if (mainAll.data.Elocs[i][0] == 152):
#         print (i, mainAll.data.Elocs[i])


def genUnifLocsL23(isd=1, Nsyn=1920, locSeg=[0,1], localRandom=False):   # intersynaptic distance in microns 
    # uniform synapse density on all branches - 
    # print 'locSeg: ', locSeg[0], locSeg[1]
    
    ## first we select synapse locations on dendritic branches 
    ## sequentially with identical inter-synapse distance
    ##      so dendrites 0-1-2... will receive inputs sequentailly from assembly 1-2-3...
    locs = []
    dend_n = 0
    for dend in model.dends:
        distance = locSeg[0]*dend.L + isd
        while distance<(dend.L * locSeg[1]):
            locs.append([dend_n, distance/dend.L])
            distance = distance + isd
        dend_n = dend_n + 1

    if (len(locs) != Nsyn):
        print('error: length of locs is not the same as Nsyn', len(locs), Nsyn)
        sys.exit(1)
    
    ## second, we take groups of 40 synapses (matching the 40 assemblies)
    ##  - form sequences without clusters  
    ##  - place these sequences on contiguous dendritic segments
    newlocs = list(locs)
    randshift = np.random.randint(0, Nsyn)

    c0 = 16
    c1 = 6
    c2 = 5
    c3 = 4

    pat0 = np.mod(np.arange(start=1, stop=5*16, step=5), 16) + 1
    pat1 = np.array([1,3,5,2,4,6])
    pat2 = np.array([1, 4, 2, 5, 3])
    pat3 = np.array([1,3,2,4])

    cc0 = np.concatenate([np.tile(pat0, 15), 
        np.roll(np.tile(pat0, 15), 3), 
        np.roll(np.tile(pat0, 15), 6), 
        np.roll(np.tile(pat0, 15), 1), 
        np.roll(np.tile(pat0, 15), 4), 
        np.roll(np.tile(pat0, 15), 7), 
        np.roll(np.tile(pat0, 15), 2), 
        np.roll(np.tile(pat0, 15), 5)])
    cc1 = np.tile(pat1, c0*c2*c3)
    cc2 = np.tile(pat2, c0*c1*c3)
    cc3 = np.tile(np.concatenate([np.tile(pat3, c1*c2/2), np.roll(np.tile(pat3, c1*c2/2), -1)]), c0)

    kk_mult = np.array([c1*c2*c3, c2*c3, c3, 1])
    cc_bind = np.vstack((cc0-1, cc1-1, cc2-1, cc3))
    kk_o = np.matmul(kk_mult, cc_bind)
    kk = np.roll(kk_o, randshift) - 1

    for isyn in np.arange(Nsyn):
        newlocs[kk[isyn]] = locs[isyn]

    ## we randomize synapses locally within each branch
    if (localRandom == True):
        el = np.asarray(newlocs)
        idend_all = el[:,0]
        locsyn_all = el[:,1]

        dends_with_syn = np.unique(idend_all)

        for idend in dends_with_syn:
            indsyn_d = np.where(idend_all == idend)[0]
            L = len(indsyn_d)
            locsyn_d = np.random.uniform(size=L)
            for isyn in range(L):
                newlocs[indsyn_d[isyn]] = [idend, locsyn_d[isyn]]


    return newlocs # list of [dend distance] pairs

# for i in np.arange(2000):
#     if (mainAll.data.Elocs[i][0] == 152):
#         print (i, mainAll.data.Elocs[i])

def genAllLocs(isd=1, locSeg=[0,1]):   # intersynaptic distance in microns 
    # uniform synapse density on all branches - 
    # print 'locSeg: ', locSeg[0], locSeg[1]
    locs = []
    dend_n = 0
    for dend in model.dends:
        distance = locSeg[0]*dend.L + isd
        while distance<(dend.L * locSeg[1]):
            locs.append([dend_n, distance/dend.L])
            distance = distance + isd
        dend_n = dend_n + 1
    return locs # list of [dend distance] pairs

def genClustStarts(Nsyn, Nclust, Ncell_per_clust, seed):
    np.random.seed(seed)
    Nsyn_in_clust = Nclust * Ncell_per_clust# number of synapses in clusters
    Nsyn_rand = Nsyn - Nsyn_in_clust
    clStarts = np.sort(np.random.choice(Nsyn_rand, Nclust, replace=False))
    clStarts = clStarts + np.arange(0, Nsyn_in_clust, Ncell_per_clust)
    return clStarts

def genClustLocs(nsyn, Nclust, Ncell_per_clust, seed, midle=False):
    # Nclust clusters and random background innervation
    # cluster locations are SELECTED from a random background
    # clustering: 
    #     1. select pre clusters by genClustStarts
    #     2. select post cluster locations by genClustStarts
    #     3. generate ordered Elocs
    #     4. remove post cluster locations from Elocs
    #     5. cycle through all clusters and add the post from either the random or the post cluster locations
    # output: Elocs: list with [dend id, location]

    # 1. presynaptic partners belonging to the clusters
    if midle == True :
        Nsyn_in_clust = Nclust * Ncell_per_clust# number of synapses in clusters
        Nsyn_rand = nsyn - Nsyn_in_clust
        istart = int(Nsyn_rand / 2)
        iend = int(Nsyn_rand / 2) + Nsyn_in_clust
        prestarts = np.arange(istart, iend, Ncell_per_clust)
    else:
        prestarts = genClustStarts(nsyn, Nclust, Ncell_per_clust, 100 * seed)

    # index of pre cells IN clusters
    ind_clustpre = np.arange(prestarts[0], prestarts[0]+Ncell_per_clust)
    if Nclust > 0:
        for j in np.arange(1, Nclust):
            istart = prestarts[j]
            iend = prestarts[j] + Ncell_per_clust
            ind_clustpre = np.concatenate((ind_clustpre, np.arange(istart, iend)))
    
    # print ind_clustpre

    # 2. synapses belonging to the clusters
    poststarts = genClustStarts(nsyn, Nclust, Ncell_per_clust, 100 * seed + 1)

    # index of synapses NOT in clusters
    ind_randpost = np.arange(poststarts[0])
    if Nclust > 1:
        for j in np.arange(1, Nclust):
            istart = poststarts[j-1] + Ncell_per_clust
            iend = poststarts[j]
            ind_randpost = np.concatenate((ind_randpost, np.arange(istart, iend)))
    else :
        j = 0
    istart = poststarts[j] + Ncell_per_clust
    iend = nsyn
    ind_randpost = np.concatenate((ind_randpost, np.arange(istart, iend)))
    print(len(ind_randpost))

    # shuffling the post clusters
    np.random.seed(100*seed + 2)
    np.random.shuffle(poststarts)

    # index of synapses IN clusters
    ind_clustpost = np.arange(poststarts[0], poststarts[0]+Ncell_per_clust)
    if Nclust > 1:
        for j in np.arange(1, Nclust):
            istart = poststarts[j]
            iend = poststarts[j] + Ncell_per_clust
            ind_clustpost = np.concatenate((ind_clustpost, np.arange(istart, iend)))
    print(len(ind_clustpost))

    # 3. random locations - all synapses
    ordered_Elocs = genRandomLocs(nsyn, seed=100*seed+3)

    # 4. random locations - NOT in clusters
    random_Elocs = [ ordered_Elocs[i] for i in ind_randpost]
    np.random.seed(100*seed + 4)
    np.random.shuffle(random_Elocs)

    # random locations - IN clusters
    clustered_Elocs = [ ordered_Elocs[i] for i in ind_clustpost]
    
    # 5. cycle through all clusters and add the post from either the 
    #       random or the post cluster locations
    j = 0 # POST - clustered
    k = 0 # POST - random
    Elocs = ordered_Elocs
    for i in np.arange(nsyn): # PRE
        if i in ind_clustpre: # take it from the cluster - list
            Elocs[i] = clustered_Elocs[j]
            j = j + 1
        else :
            Elocs[i] = random_Elocs[k]
            k = k + 1
    print(len(Elocs))

    return Elocs, ind_clustpre

def addClustLocs(nsyn, Nclust, Ncell_per_clust, seed, midle=False, clocs=None, Lmin=60):
    # Nclust clusters in a random background innervation
    # clusters are ADDED to the random background
    # clustering: 
    #     1. select pre clusters by genClustStarts
    #     2. generate random Elocs
    #     3. select post cluster locations by genClusts
    #     4. add post cluster locations
    # output: Elocs: list with [dend id, location]
    
    # 1. presynaptic partners belonging to the clusters
    if midle == True : # clusters start at the middle of the maze
        Nsyn_in_clust = Nclust * Ncell_per_clust# number of synapses in clusters
        Nsyn_rand = nsyn - Nsyn_in_clust
        if data.model == 'CA1':
            istart = 880 #int(Nsyn_rand / 2)
            iend = 1120 # int(Nsyn_rand / 2) + Nsyn_in_clust
            # istart = int(Nsyn_rand / 2)
            # iend = int(Nsyn_rand / 2) + Nsyn_in_clust
        elif data.model == 'L23' :
            istart = int(nsyn / 16 * 13)
            iend = int(nsyn / 16 * 13) + Nsyn_in_clust
            # istart = int(Nsyn_rand / 2)
            # iend = int(Nsyn_rand / 2) + Nsyn_in_clust
        prestarts = np.arange(istart, iend, Ncell_per_clust)        
    else:
        prestarts = genClustStarts(nsyn, Nclust, Ncell_per_clust, 100 * seed)

    # print(prestarts)
    # index of pre cells IN clusters
    ind_clustpre = np.arange(prestarts[0], prestarts[0]+Ncell_per_clust)
    if Nclust > 0:
        for j in np.arange(1, Nclust):
            istart = prestarts[j]
            iend = prestarts[j] + Ncell_per_clust
            ind_clustpre = np.concatenate((ind_clustpre, np.arange(istart, iend)))

    # 2. random locations - background synapses
    if (Nsyn_rand): 
        bg_Elocs = genRandomLocs(Nsyn_rand, seed=100*seed+1)
        np.random.seed(100*seed + 2)
        np.random.shuffle(bg_Elocs)
    else:
        bg_Elocs = []

    # 3. clustered locations
    # def genClusts(Nclust, Ncell_per_clust, minL, seed):
    clustered_Elocs = genClusts(Nclust, Ncell_per_clust, Lmin, seed=100*seed+4, clocs=clocs)
    # clustered_Elocs = genClusts(Nclust, Ncell_per_clust, 1, seed=100*seed+4)
    
    # 4. cycle through all clusters and add the post from either the 
    #       random or the post cluster locations
    j = 0 # POST - clustered
    k = 0 # POST - random
    Elocs = bg_Elocs + clustered_Elocs
    for i in np.arange(nsyn): # PRE
        if i in ind_clustpre: # take it from the cluster - list
            Elocs[i] = clustered_Elocs[j]
            j = j + 1
        else :
            Elocs[i] = bg_Elocs[k]
            k = k + 1
    print(len(Elocs))

    return Elocs, ind_clustpre


def rearrangeBgLocs(Elocs, DSyn, iclust, ibg):
    synDist = np.array(DSyn, copy=True)
    mDsyn = np.amax(synDist)

    # this vector will contain the unique index pointing to the location of closest 
    # synapse to the clustered synapses
    peer_of_clust = np.zeros(len(Elocs)) 

    # this vector will contain the identitiy of the already booked random synapses
    # this is just a sanity check - should sum to 240
    booked_rand = np.zeros(len(Elocs))

    # this list will contain the identitiy of the already booked background synapses
    booked_bg = []

    for i_cell in iclust:
        dd = synDist[i_cell,]
        dd[iclust] = max(dd)
        i_peer = np.argmin(dd)
        
        peer_of_clust[i_cell] = i_peer
        synDist[:,i_peer] = mDsyn

        booked_rand[i_peer] = 1
        if (i_peer in ibg):
            booked_bg.append(i_peer)

    if sum(booked_rand) != 240:
        print('not all peers have been found')
        return

    ## we need the index of those bg inputs that are not yet assigned to clustered synapses
    free_bg = np.setdiff1d(ibg, booked_bg)

    ### now check that the peer is B(ackground, > 2000)
    ### if not, switch it with a B that is not peer
    for i_cell in iclust:
        peer_index = int(peer_of_clust[i_cell])
        if (peer_index not in ibg):
            ## the new bg is sampled from the free_bg 
            new_peer = np.random.choice(free_bg)
            free_bg = np.setdiff1d(free_bg, new_peer)
            loc_new = Elocs[peer_index]
            Elocs[peer_index] = Elocs[new_peer]
            Elocs[new_peer] = loc_new

    return Elocs


def addBgLocs(Elocs, iclust):

    for i_cell in iclust:
        Elocs.append(Elocs[i_cell])
    return Elocs


#-----------------------------------------------
# Input generation functions
#-----------------------------------------------

def genPoissonTrain(Ensyn, Erate, Insyn=0, Irate = 0, duration=10, random_seed=1):
    ## Poisson random spike train with Erate and Irate
    ## rates are in Hz
    ## duration: total simulation time in s
    ## times: two columns: cell id, spike times (ms)
    np.random.seed(random_seed)
    Etimes = np.array([])
    if (Insyn > 0) : Itimes = np.array([])

    while (Etimes.shape[0]<2): # at least two spikes are required
        Etimes = np.array([[0,0]])
        if (Insyn > 0) : Itimes = np.array([[0,0]])
        state = 0
        PE =  br.OfflinePoissonGroup(Ensyn, Erate, duration * br.ms)
        Etimes = np.array(PE.spiketimes)
        if (Etimes.shape[0] > 0):
            Etimes[:,1] = Etimes[:,1] * 1000
        if ((Insyn > 0)&(Irate > 0)) :
            PI =  br.OfflinePoissonGroup(Insyn, Irate, duration * br.ms)
            Itimes = np.array(PI.spiketimes)
            if (Itimes.shape[0] > 0):
                Itimes[:,1] = Itimes[:,1] * 1000
        if (Insyn > 0) :
            times = [Etimes, Itimes]
        else :
            times = Etimes
    return times

def readTrain(type, Ensyn, Insyn, Erate, Irate, duration, stimseed, rep):

    ## eitimes = readPoissonTrain(duration, ir, rep, Ensyn, bgErate, fgErate, Insyn, bgIrate, fgIrate)
    ## rates are in Hz
    ## duration: total simulation time in s
    ## times: two columns: cell id, spike times (ms)
    Etimes = np.array([])
    print('read Train: duration: ', duration)

    fname = './'+str(data.model)+'stims/'+str(type)+'/Espikes_d'+str(duration)+'_Ne'+str(Ensyn)+'_Re'+str(Erate)+'_rseed'+str(stimseed)+'_rep'+str(rep)+'.dat'
    print('loading input from file', fname)
    Etimes = loadtxt(fname, comments="#", delimiter=" ", unpack=False)    
    
    if (Insyn > 0) :
        Itimes = np.array([])
        fname = './'+str(data.model)+'stims/'+str(type)+'/Ispikes_d'+str(duration)+'_Ni'+str(Insyn)+'_Ri'+str(Irate)+'_rseed'+str(stimseed)+'_rep'+str(rep)+'.dat'
        Itimes = loadtxt(fname, comments="#", delimiter=" ", unpack=False)    
        times = [Etimes, Itimes]
    else :
        times = Etimes

    return times


def readReplayTrain(type, Ensyn, Insyn, Erate, Irate, duration, startloc, rep):

    ## eitimes = readPoissonTrain(duration, ir, rep, Ensyn, bgErate, fgErate, Insyn, bgIrate, fgIrate)
    ## rates are in Hz
    ## duration: total simulation time in s
    ## times: two columns: cell id, spike times (ms)
    Etimes = np.array([])
    print('read Train: duration: ', duration)
    
    fname = './'+str(data.model)+'stims/SPW/'+str(type)+'/Espikes_d03_Ne'+str(Ensyn-240)+'_Re'+str(Erate)+'_dstart'+str(startloc)+'_rep'+str(rep)+'.dat'
    print('loading input from file', fname)
    Etimes = loadtxt(fname, comments="#", delimiter=" ", unpack=False)    
    
    if (Insyn > 0) :
        Itimes = np.array([])
        fname = './'+str(data.model)+'stims/SPW/'+str(type)+'/Ispikes_d03_Ni'+str(Insyn)+'_Ri'+str(Irate)+'_dstart'+str(startloc)+'_rep'+str(rep)+'.dat'
        Itimes = loadtxt(fname, comments="#", delimiter=" ", unpack=False)    
        times = [Etimes, Itimes]
    else :
        times = Etimes

    return times



def genDSinput(nsyn, Nmax, tInterval, onset, direction='OUT'):
    ## a single train with nsyn inputs - either in the in or in the out direction
    times = np.zeros([nsyn, 2])
    if (direction=='OUT'):
        times[:,0] = np.arange(0, nsyn)
    else:
        times[:,0] = np.arange(Nmax-1, Nmax-nsyn-1, -1)
    # print(times)
    # tt = np.arange(0, nsyn*tInterval, tInterval) + onset
    # print(tt)
    times[:,1] = np.arange(0, nsyn*tInterval, tInterval)[0:nsyn] + onset
    return times


#--------------------------------------------------
# Simulation functions
#--------------------------------------------------

def sim_PoissonInput(Ensyn, Erate,  Insyn, Irate, duration, rand_seed=1):
    ## Poisson spike train - Erate [Irate] is the rate of the individual input neuronss
    eitimes = genPoissonTrain(Ensyn, Erate, Insyn, Irate, duration*1000, rand_seed)
    if ((Insyn > 0) & (Irate > 0)) :
        etimes = eitimes[0]
        itimes = eitimes[1]
        print(len(etimes[:,0]), 'E spikes generated', len(itimes[:,0]), 'I spikes generated')
        data.itimes = itimes
    else :
        etimes = eitimes
        print(len(etimes[:,0]), 'E spikes generated')
        itimes = None
        data.itimes = itimes
    data.etimes = etimes
 
    # Run
    fih = lb.h.FInitializeHandler(1, initSpikes)
    taxis, v, vD, Gdend = lb.simulate(model, t_stop=duration * 1000, NMDA=data.NMDA, recDend=data.recordDend, i_recDend=data.locDendRec)

    return taxis, v, vD, etimes, itimes

# def readTrain(type, duration, Ensyn, Insyn, Erate, Irate, stimseed, rep):

def sim_PlaceInput(type, Ensyn, Insyn, Erate, Irate, duration, stimseed, rep, recNMDA, elimIspike=0):
    ## spike train read from file - Erate [Irate] is the rate of the individual input neuronss

    print('sim Place Input: duration: ', duration)
    eitimes = readTrain(type, Ensyn, Insyn, Erate, Irate, duration, stimseed, rep)
    if ((Insyn > 0) & (Irate > 0)) :
        etimes = eitimes[0]
        itimes = eitimes[1]
        #### this is a hack to phase shift inhibitory spikes
        # itimes[:,1] = (itimes[:,1] + 115) % max(itimes[:,1])
        print(len(etimes[:,0]), 'E and ',  len(itimes[:,0]),  'I spikes read from file')
        if (elimIspike > 0):
            N_ispikes = len(itimes)
            i_index = np.sort(np.random.choice(N_ispikes, int(round((1-elimIspike) * N_ispikes)), replace=False))
            itimes = itimes[i_index]
            print('number of I spikes reduced to', len(itimes))
        data.itimes = itimes
    else :
        etimes = eitimes
        print(len(etimes[:,0]), 'E spikes generated')
        itimes = None
        data.itimes = itimes

    data.etimes = etimes
 
    # Run
    fih = lb.h.FInitializeHandler(1, initSpikes)
    if (data.SPINES):
        recSpines = data.locSpineRec
    else :
        recSpines = []

    taxis, v, vD, Gdend, Idend = lb.simulate(model, t_stop=data.TSTOP * 1000, NMDA=data.NMDA, recDend=data.recordDend, i_recDend=data.locDendRec, x_recDend=data.xDendRec, spines=data.SPINES, i_recSpine=recSpines, recGDend=True, i_recSyn=recNMDA)
    # taxis, v, vD = lb.simulate(model, t_stop=data.TSTOP * 1000, NMDA=data.NMDA, recDend=data.recordDend, i_recDend=data.locDendRec)

    return taxis, v, vD, etimes, itimes, Gdend, Idend

def sim_ReplayInput(type, Ensyn, Insyn, Erate, Irate, duration, startloc, rep, recNMDA):
    ## spike train read from file - Erate [Irate] is the rate of the individual input neuronss
    print('sim Place Input: duration: ', duration)
    eitimes = readReplayTrain(type, Ensyn, Insyn, Erate, Irate, duration, startloc, rep)

    if ((Insyn > 0) & (Irate > 0)) :
        etimes = eitimes[0]
        itimes = eitimes[1]
        #### this is a hack to phase shift inhibitory spikes
        # itimes[:,1] = (itimes[:,1] + 115) % max(itimes[:,1])
        print(len(etimes[:,0]), 'E and ',  len(itimes[:,0]),  'I spikes read from file')
        data.itimes = itimes
    else :
        etimes = eitimes
        print(len(etimes[:,0]), 'E spikes generated')
        itimes = None
        data.itimes = itimes

    data.etimes = etimes
 
    # Run
    fih = lb.h.FInitializeHandler(1, initSpikes)
    if (data.SPINES):
        recSpines = data.locSpineRec
    else :
        recSpines = []

    taxis, v, vD, Gdend, Idend = lb.simulate(model, t_stop=data.TSTOP * 1000, NMDA=data.NMDA, recDend=data.recordDend, i_recDend=data.locDendRec, x_recDend=data.xDendRec, spines=data.SPINES, i_recSpine=recSpines, recGDend=True, i_recSyn=recNMDA)
    # taxis, v, vD = lb.simulate(model, t_stop=data.TSTOP, NMDA=data.NMDA, recDend=data.recordDend, i_recDend=data.locDendRec)

    return taxis, v, vD, etimes, itimes, Gdend, Idend

# --------------------------------------------------
# Iteration functions
# --------------------------------------------------

def SIM_PoissonIteration(Ensyn, Erate,  Insyn, Irate, duration, nIter):
    # nIter simulation with same paramaters but different random seed
    # rates are in Hz, duration in s
    for iter in np.arange(nIter):
        r_seed = 100 + 4 * iter
        print('Running E rate', Erate, 'I rate', Irate, 'iteration:', iter)
        data.taxis, v, vD, Et, It = sim_PoissonInput(Ensyn=Ensyn, Erate=Erate, Insyn=Insyn, Irate=Irate, duration=duration, rand_seed=r_seed)
        storeSimInputOutput(v,vD,Et,It)

#def sim_PlaceInput(type, Ensyn, Insyn, Erate, Irate, duration, stimseed, rep):

def SIM_PlaceIteration(type, Ensyn, Insyn, Erate, Irate, duration, Nrep, stimseed, recNMDA, elimIspike=1):
    # Nrep simulation with same paramaters but different input spikes read from file
    # rates are in Hz, duration in s
    
    for iter in np.arange(Nrep):
    # iter = 11
        print('Running iteration', iter, 'duration: ', duration)
        data.taxis, v, vD, Et, It, gN, iN = sim_PlaceInput(type=type, Ensyn=Ensyn, Insyn=Insyn, Erate=Erate, Irate=Irate, duration=duration, stimseed=stimseed, rep=iter, recNMDA=recNMDA, elimIspike=elimIspike)
        # storeSimInputOutputGdendIdend(v,vD,Et,It,gN,iN)
        # data.taxis, v, vD, Et, It = sim_PlaceInput(type=type, Ensyn=Ensyn, Insyn=Insyn, Erate=Erate, Irate=Irate, duration=duration, stimseed=stimseed, rep=iter, recNMDA=recNMDA)
        # storeSimInputOutput(v,vD,Et,It)
        storeSimInputOutputGdendIdend(v,vD,Et,It,gN,iN)


def SIM_ReplayIteration(type, Ensyn, Insyn, Erate, Irate, duration, Nrep, recNMDA):
    # Nrep simulation with same paramaters but different input spikes read from file
    # rates are in Hz, duration in s
    for iter in np.arange(Nrep):
        k = 0
        for startdist in np.arange(0, 200, 10):
        # for startdist in [20, 100]:
            print('Running iteration', iter, 'duration: ', duration)
            taxis_i, v_i, vD_i, Et_i, It_i, gN_i, iN_i = sim_ReplayInput(type=type, Ensyn=Ensyn, Insyn=Insyn, Erate=Erate, Irate=Irate, duration=duration, startloc=startdist, rep=iter, recNMDA=recNMDA)
            if (k == 0):
                # synDist = np.array(DSyn, copy=True)
                taxis = np.array(taxis_i, copy=True)
                v = np.array(v_i, copy=True)
                vD = np.array(vD_i, copy=True)
                gN = np.array(gN_i, copy=True)
                iN = np.array(iN_i, copy=True)
                Et = np.array(Et_i, copy=True)
                It = np.array(It_i, copy=True)
            else :
                taxis = np.concatenate([taxis, taxis_i + (duration*1000 + 100) * k])
                v = np.concatenate([v, v_i])
                vD = np.column_stack([vD, vD_i])
                gN = np.column_stack([gN, gN_i])
                iN = np.column_stack([iN, iN_i])
                Et_i[:,1] = Et_i[:,1] + (duration*1000 + 100) * k
                It_i[:,1] = It_i[:,1] + (duration*1000 + 100) * k
                Et = np.row_stack([Et, Et_i])
                It = np.row_stack([It, It_i])
            k = k + 1

        data.taxis = taxis
        storeSimInputOutputGdendIdend(v,vD,Et,It,gN,iN)


def SIM_nsynIteration(maxNsyn, tInterval, onset, direction='OUT'):
# direction should be either 'OUT' or 'IN' (default)
#   ---------------------------------------
#   first each synapse is activated alone
    It = None
    # maxNsyn = sum(maxNsyn)
    print(maxNsyn)
    for nsyn in np.arange(1, maxNsyn+1): # 1 - N
    # for nsyn in [1]: # 1 - N
        print('activating synapse ', nsyn, ' alone')
        if (direction=='OUT'):
            data.etimes = np.array([[nsyn-1, onset * 1000]]) # 0 - N-1
        else:
            data.etimes = np.array([[maxNsyn - nsyn, onset * 1000]]) # N-1 - 0

        fih = lb.h.FInitializeHandler(1, initSpikes)

        # def simulate(model, t_stop=100, NMDA=False, recDend=False, i_recDend=11, x_recDend=0.5, spines=False, i_recSpine=11):
        taxis, v, vD, Gdend, Idend = lb.simulate(model, t_stop=data.TSTOP * 1000,
                                         NMDA=data.NMDA, recDend=data.recordDend,
                                         i_recDend=data.locDendRec, x_recDend=data.xDendRec, 
                                         recGDend = data.recordGDend)
        Et = data.etimes
        storeSimInputOutputGdend(v, vD, Et, It, Gdend)

#   ---------------------------------------
#   next, synapses are activated together
    for nsyn in np.arange(1, maxNsyn+1):
    # for nsyn in [10, 20, 30]:
        print('activating ', nsyn, ' synapses together')
        data.etimes = genDSinput(nsyn, maxNsyn, tInterval, onset * 1000, direction)
        fih = lb.h.FInitializeHandler(1, initSpikes)
        taxis, v, vD, Gdend, Idend = lb.simulate(model, t_stop=data.TSTOP * 1000,
                                         NMDA=data.NMDA, recDend=data.recordDend,
                                         i_recDend=data.locDendRec, x_recDend=data.xDendRec,
                                         recGDend = data.recordGDend)
        Et = data.etimes
        storeSimInputOutputGdend(v, vD, Et, It, Gdend)
    data.taxis = taxis


def SIM_currentSteps(iRange):
    Et, It = [], []
    for step in iRange:
        print('Running input step current', step)
        model.stim.amp = step
        taxis, v, vD, GDend = lb.simulate(model, t_stop=data.TSTOP * 1000, NMDA=data.NMDA, recDend=data.recordDend, i_recDend=data.locDendRec, x_recDend=data.xDendRec)
        storeSimOutput(v, vD)
    data.taxis = taxis


def SIM_minisDistribution(data):
#   ---------------------------------------
#   first each excitatory synapse is activated alone
    It, Et, data.etimes, data.itimes = [], [], [], []
    Enum = data.Ensyn
    for nsyn in np.arange(0, Enum):
    # for nsyn in [23, 952, 981]:
        print('activating excitatory synapse ', nsyn, ' alone')
        data.etimes = np.array([[nsyn, data.st_onset * 1000]])
        fih = lb.h.FInitializeHandler(1, initSpikes)
        taxis, v, vD, GDend = lb.simulate(model, t_stop=data.TSTOP * 1000,
                                         NMDA=data.NMDA, recDend=data.recordDend,
                                         i_recDend=data.locDendRec, x_recDend=data.xDendRec)
        Et = data.etimes
        storeSimOutput(v, vD)

#   ---------------------------------------
#   next, each inhibitory synapse is activated alone
    It, Et, data.etimes, data.itimes = [], [], [], []
    Inum = data.Insyn
    for nsyn in np.arange(0, Inum):
        print('activating inhibitory synapse ', nsyn, ' alone')
        data.itimes = np.array([[nsyn, data.st_onset * 1000]])
        fih = lb.h.FInitializeHandler(1, initSpikes)
        taxis, v, vD, GDend = lb.simulate(model, t_stop=data.TSTOP * 1000,
                                         NMDA=data.NMDA, recDend=data.recordDend,
                                         i_recDend=data.locDendRec, x_recDend=data.xDendRec)
        It = data.itimes
        storeSimOutput(v, vD)

    data.taxis = taxis





