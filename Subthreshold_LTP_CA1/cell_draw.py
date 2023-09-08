import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm

from neuron import h

def rotate(x,z,theta):
    x2 =  x * np.cos(theta) + z * np.sin(theta)
    z2 =  z * np.cos(theta) - x * np.sin(theta)
    return [x2, z2]

def plot_syns(data, model, plotInh=False, iB=[]):
    #----------------------------------------------------------------------------
    # Some parameters
    nsyn = data.Ensyn
    if (plotInh): nsyn = int(nsyn + data.Insyn)
    #----------------------------------------------------------------------------
    # Create neuron and generate locations
    cell_secs = []
    for sec in h.allsec(): 
        cell_secs.append(sec)
        # sec.nseg = 100
    locs = np.array(data.Elocs)
    if (plotInh): locs = np.vstack((locs, np.array(data.Ilocs)))
    #----------------------------------------------------------------------------
    # Add dummy compartments at synapse locations
    synapses = []
    print ('number of synapses:', nsyn)
    for s in np.arange(0,nsyn):
        synapses.append(h.Section())
        synapses[s].L = 5# 1e-6
        synapses[s].diam = 1#1e-6
        idend = np.int(locs[s,0])
        if (idend > -1):
            synapses[s].connect(model.dends[idend], np.int(locs[s,1]), 0)
        else :
            synapses[s].connect(model.soma, 0.5, 0)
    # ----------------------------------------------------------------------------
    # Get numbers for plotting cell morphology
    # cell: [xstart xend ystart yend diamstart diamend]
    # synapse: [x, y]
    h.define_shape()
    cell_coordinates = []
    k = 0
    theta = 0 # -np.pi/2
    for sec in cell_secs:
        sec.push()
        for stepCount in np.arange(1, h.n3d()):
            stepCount =  float(stepCount)
            rot_xz_m1 = rotate(h.x3d(stepCount-1), h.z3d(stepCount-1), theta)
            rot_xz = rotate(h.x3d(stepCount), h.z3d(stepCount), theta)
            cell_coordinates.append([rot_xz_m1[0], rot_xz[0],
                                 h.y3d(stepCount-1), h.y3d(stepCount),
                                     h.diam3d(stepCount-1), h.diam3d(stepCount)])
        h.pop_section()
        k = k + 1
    print('number of sections processed:', k)


    set_dends = set(locs[:,0])
    # I don't know how to work with sets, so convert it to an array
    set_dends = np.array(list(set_dends))
    dend_coord = []
    for i_dend in np.arange(0,len(set_dends)):
        syn_coordinates = []
        d_dend = np.int(set_dends[i_dend])
        if (d_dend > -1):
            model.dends[d_dend].push()
        else :
            model.soma.push()
        for stepCount in np.arange(1, h.n3d()):
            stepCount =  float(stepCount)
            rot_xz_m1 = rotate(h.x3d(stepCount-1), h.z3d(stepCount-1), theta)
            rot_xz = rotate(h.x3d(stepCount), h.z3d(stepCount), theta)
            syn_coordinates.append([rot_xz_m1[0], rot_xz[0],
                                    h.y3d(stepCount-1), h.y3d(stepCount),
                                    h.diam3d(stepCount-1), h.diam3d(stepCount)])
        h.pop_section()
        dend_coord.append(syn_coordinates)

    #----------------------------------------------------------------------------
    # Make plot
    plt.figure(figsize=(8,8))
    # Cell
    for pt in np.arange(len(cell_coordinates)):
        xstart, xend = cell_coordinates[pt][0], cell_coordinates[pt][1]
        ystart, yend = cell_coordinates[pt][2], cell_coordinates[pt][3]
        lx = xend-xstart
        ly = yend-ystart
        l = np.sqrt(np.dot([lx,ly],[lx,ly]))
        diamstart = cell_coordinates[pt][4] / 2
        diamend = cell_coordinates[pt][5] / 2

        # if diamstart > 8: diamstart=diamstart/3.5
        # if diamend > 8: diamend=diamend/3.5

        if l>0:
            plt.fill([xstart+2*diamstart*ly/(2*l), xend+2*diamend*ly/(2*l),
                     xend-2*diamend*ly/(2*l), xstart-2*diamstart*ly/(2*l)], 
                     [ystart-2*diamstart*lx/(2*l), yend-2*diamend*lx/(2*l),
                     yend+2*diamend*lx/(2*l), ystart+2*diamstart*lx/(2*l)],
                     'k', lw=0) 

        c1 = plt.Circle((xstart, ystart), diamstart, color='k', lw=0)
        c2 = plt.Circle((xend, yend), diamend, color='k', lw=0) 
        plt.gca().add_patch(c1)
        plt.gca().add_patch(c2)
        
    # Synapses
    cmap = plt.cm.get_cmap('gist_rainbow')  # color=cmap(i_rep/float(nReps)) 
    ecols = ['#ffcc99', '#ccff99']     
    ccols = ['#ff0000', '#ff3399', '#33ff99', '#ff0000']     # clusters
    icols = ['#99ccff']

    col40 = ['#ddd9e0', '#d4d6dc', '#c2ced4', '#afc4cd', 
        '#9cb9c8', '#8aaec5', '#7ca2c2', '#7195c0', 
        '#6989be', '#647dbc', '#616fb9', '#5f5fb3', 
        '#5e52ad', '#5d42a4', '#5b3298', '#572487', 
        '#501a74', '#471461', '#3d114d', '#34113e', 
        '#311337', '#3b113b', '#481341', '#571647', 
        '#691a4d', '#792050', '#892950', '#953250', 
        '#a03e50', '#aa4a50', '#b35752', '#ba6557', 
        '#c0765f', '#c5866a', '#c99578', '#cca186', 
        '#d0b29c', '#d5c0b2', '#dbccc8', '#dfd4d7']


    col16 = ['#d5d6dc','#a9c1cb','#7ea4c3','#6684bd',
    '#6684bd','#5f5eb3','#5c379c','#4d186d',
    '#35113f','#41123d','#691a4d','#922f50',
    '#ac4d51','#c0745d','#cb9d82','#d5c0b2']


    col6 = ['#d5d6dc','#80a5c3','#5f58b0','#321237','#983550','#c9977b']

    col5 = ['#d5d6dc', '#6e92c0', '#5c379c', '#882850', '#c89174', '#e1d8de']

    col4 = ['#d5d6dc', '#6684bd', '#311337', '#b25652']

    for syn in np.arange(nsyn):
        # select the branch for the synapse
        idend = np.int(locs[syn,0])
        iidend = np.int(np.argwhere(set_dends == idend))
        syn_coordinates = dend_coord[iidend]
        ptmax=np.shape(syn_coordinates)[0]
        # find the coordinates along the branch
        ptt = locs[syn,1] * ptmax
        pt = np.int(np.floor(ptt))
        ptd = ptt - pt
        xstart, xend = syn_coordinates[pt][0], syn_coordinates[pt][1]
        ystart, yend = syn_coordinates[pt][2], syn_coordinates[pt][3]
        lx = xend-xstart
        ly = yend-ystart
        x = xstart# + lx * ptd # this correction works when complex 3D info is plotted
        y = ystart# + lx * ptd
        if (syn < data.Ensyn):
            if not (syn in data.ind_clust):
                if syn in iB: 
                    ptcol = ecols[1]
                else:
                    ptcol = ecols[0]                    
                # ptcol = cmap((syn) / 40 / 40.0)
                c = plt.Circle((x,y), 3, color=ptcol, lw=2)
                plt.gca().add_patch(c)   
        else :
            # print('nsyn:', syn, 'idend:', idend, 'location:', locs[syn,1])
            c = plt.Circle((x,y), 3, color=icols[0], lw=2)
            plt.gca().add_patch(c)   


    for syn in data.ind_clust:
    # for syn in np.arange(300):
        # select the branch for the synapse
        idend = np.int(locs[syn,0])
        iidend = np.int(np.argwhere(set_dends == idend))
        syn_coordinates = dend_coord[iidend]
        ptmax=np.shape(syn_coordinates)[0]
        # find the coordinates along the branch
        ptt = locs[syn,1] * ptmax
        pt = np.int(np.floor(ptt))
        x = syn_coordinates[pt][0]
        y = syn_coordinates[pt][2]
        # c = plt.Circle((x,y), 3, color=ecols[0], lw=2)
        ptcol = ccols[0]
        # CA1, colors according to place preference
        # ptcol = cmap((syn) / 50 / 40.0) # plotting synapses of place cells
        # ptcol = col40[(syn) / 50] # plotting synapses of place cells
        # L23, colors according to orientation preference
        # ptcol = col16[(syn) / 120] 
        # L23, colors according to phase preference
        # ptcol = col6[np.mod(syn, 120) / 20] # plotting synapses of place cells
        # L23, colors according to SDof orientation preference
        # ptcol = col5[np.mod((syn), 20) / 4] # plotting synapses of place cells
        # L23, colors according to SD of phase preference
        # ptcol = col4[np.mod(syn, 4)] # plotting synapses of place cells



        # print((syn) / 50 / 40.0)
        c = plt.Circle((x,y), 3, color=ptcol, lw=0)
        plt.gca().add_patch(c)   


    plt.axis('equal')
    ## CA1 - all
    plt.xlim(-250,250)
    plt.ylim(-200,600)
    ## CA1 - distal
    # plt.xlim(10,310)
    # plt.ylim(270,570)
    ## L23 - distal
    # plt.xlim(-170,0)
    # plt.ylim(50,220)
    # L23 - all
    # plt.xlim(-220,200)
    # plt.ylim(-150,250)
    plt.show()
            
