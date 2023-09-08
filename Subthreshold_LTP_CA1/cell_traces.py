import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# Plotting functions
# plot the dendritic and somatic response 
# - different trials on the same subplot
# - different compartments on different subplots

def plotTraces(data, xmax=1000, ndend=0):
    plt.figure(figsize=(8,3))
    cols = ['b', 'g', 'r', 'c']
    ax = plt.subplot(1, ndend+1, 1)
    plt.title('somatic voltage')
    nReps = len(data.vdata)
    cmap = plt.cm.get_cmap('jet')   

    for i_rep in np.arange(0,nReps):
        trace = data.vdata[i_rep]
        ax.plot(data.taxis, trace, color=cmap(i_rep/float(nReps)))
        plt.xlim(0,xmax)
        plt.ylim(-75,45)

    if (ndend > 0):
        for i_den in np.arange(ndend):
            ax = plt.subplot(1, ndend+1, i_den+2)
            if data.recordDend:
                for i_rep in np.arange(0,nReps):
                    trace = data.vDdata[i_rep]
                    Dtrace = trace[i_den]
                    ax.plot(data.taxis, Dtrace, color=cmap(i_rep/float(nReps)))
                    plt.xlim(0,xmax)
                    plt.ylim(-75,45)
            subplot_title = 'dendrite' + str(data.locDendRec[i_den])
            plt.title(subplot_title)
    plt.show()


#----------------------------------------------------------------------------
# plot the dendritic and somatic response 
# - each trial is a different subplot
# - all compartments are plotted on the same subplot
def plotAPprop(data):
    plt.figure(figsize=(8,8))
    nReps = len(data.vdata)
    # ndend = np.shape(data.vDdata[0])[0]
    ndend = len(data.locDendRec)
    nRec = ndend + 1
    cmap = plt.cm.get_cmap('jet')   
    ncols = int(np.ceil(nReps / 2.0))

    for i_rep in np.arange(0,nReps):
        ax = plt.subplot(2, ncols, i_rep+1)
        trace = data.vdata[i_rep]
        Vlabels = ['soma']
        ax.plot(data.taxis, trace, color=cmap(1.0/nRec), label=Vlabels[0])
        plt.xlim(20,50)
        plt.ylim(-75,50)
        subplot_title = 'I = ' + str(data.iRange[i_rep])
        plt.title(subplot_title)
        
        Dtrace = data.vDdata[i_rep]
        for i_den in np.arange(ndend):
            Vlabels.append('dend-' + str(data.locDendRec[i_den]))
            iDtrace = Dtrace[i_den]
            ax.plot(data.taxis, iDtrace, color=cmap((i_den+2.0)/nRec), label=Vlabels[i_den+1])
        if (i_rep == 0) :
            legend = ax.legend(loc=0, shadow=False, fontsize='small')

    plt.show()


#----------------------------------------------------------------------------
# plot the dendritic and somatic V and somatic dVdt 
# - each trial is a different column
# - first row: the voltage of all compartments are plotted on the same subplot
# - second row: somatic dVdt

def plotV_dV(data):
    plt.figure(figsize=(8,8))
    nReps = len(data.vdata)
    # ndend = np.shape(data.vDdata[0])[0]
    ndend = len(data.locDendRec)
    nRec = ndend + 1
    cmap = plt.cm.get_cmap('jet')   
    Vlabels = ['soma']

    for i_rep in np.arange(0,nReps):
        ax = plt.subplot(2, nReps, i_rep+1)
        trace = data.vdata[i_rep]
        ax.plot(data.taxis, trace, color=cmap(1.0/nRec), label=Vlabels[0])
        plt.xlim(0,data.TSTOP*1000)
        plt.ylim(-80,30)
        subplot_title = 'voltage (mV)'
        plt.title(subplot_title)
        
        Dtrace = data.vDdata[i_rep]
        for i_den in np.arange(ndend):
            Vlabels.append('dend-' + str(data.locDendRec[i_den]))
            iDtrace = Dtrace[i_den]
            ax.plot(data.taxis, iDtrace, color=cmap((i_den+2.0)/nRec), label=Vlabels[i_den+1])

        if (i_rep == 0) :
            legend = ax.legend(loc=0, shadow=False, fontsize='small')


    for i_rep in np.arange(0,nReps):
        ax = plt.subplot(2, nReps, i_rep+nReps+1)
        trace = data.vdata[i_rep]
        dtrace = 10.0 * np.diff(np.hstack((trace[0], trace)))
        ax.plot(data.taxis, dtrace)
        subplot_title = 'dV/dt (mV/ms)'
        plt.title(subplot_title)
        # plt.xlim(0,data.TSTOP*1000)
        plt.xlim(90,130)
        plt.ylim(-0.2,4)

    plt.show()

#----------------------------------------------------------------------------
# plot the dendritic and somatic V and somatic dVdt 
# - each trial is a different column
# - first row: the voltage of all compartments are plotted on the same subplot
# - second row: local dendritic conductances are plotted - Na, Kdr, Kap, Kad, AMPA and NMDA

def plotV_G(data, secl):
    plt.figure(figsize=(12,8))
    nReps = len(data.vdata)
    # ndend = np.shape(data.vDdata[0])[0]
    ndend = len(data.locDendRec)
    nRec = ndend + 1
    cmap = plt.cm.get_cmap('jet')   
    diam = 1.0 / 10000.0 # 1 um in cm
    length = secl / 10000.0 # section length in cm
    Area = diam * np.pi * length # approximate area of a segment in cm2

    for i_rep in np.arange(0,nReps):
        ax = plt.subplot(2, nReps, i_rep+1)
        trace = data.vdata[i_rep]
        Vlabels = ['soma']
        ax.plot(data.taxis, trace, color=cmap(1.0/nRec), label=Vlabels[0])
        plt.xlim(90,190)
        plt.ylim(-76,0)
        
        Dtrace = data.vDdata[i_rep]
        for i_den in np.arange(ndend):
            iDtrace = Dtrace[i_den]
            Vlabels.append('dend-' + str(data.locDendRec[i_den]))
            ax.plot(data.taxis, iDtrace, color=cmap((i_den+2.0)/nRec), label=Vlabels[i_den+1])

        if (i_rep == 0) :
            legend = ax.legend(loc=0, shadow=False, fontsize='small')
            subplot_title = 'somatic and dendritic voltage (mV)'
            plt.title(subplot_title)

    Glabels = ['Na', 'Kdr', 'Ka-prox', 'KA-dist', 'AMPA', 'NMDA']
    if (data.model == 'L23'): Glabels = ['Na', 'Kv', 'Km', 'K-ca', 'AMPA', 'NMDA']
    for i_rep in np.arange(0,nReps):
        ax = plt.subplot(2, nReps, i_rep+nReps+1)
        plt.xlim(90,190)
        plt.ylim(-00.05,1)
        Gtraces = data.Gdata[i_rep]
        nG = np.shape(Gtraces)[0]
        for i_g in np.arange(nG):
            Gtrace = Gtraces[i_g]
            if (i_g < 4):
                Gtrace = Gtrace  * Area * 1e9 # convert from S/cm2 to nS
                if ((i_g < 3) & (data.model == 'L23')):
                    Gtrace = Gtrace * 1e-4 # convert from 1 pS/um2 to 1e-4 mho/cm2
            else :
                Gtrace = Gtrace  * 1e3 # convert from uS to nS

            ax.plot(data.taxis, Gtrace, color=cmap((i_g+1.0)/nG), label=Glabels[i_g])
        if (i_rep == 0) :
            legend = ax.legend(loc=0, shadow=False, fontsize='small')
            subplot_title = 'local dendritic conductances (nS)'
            plt.title(subplot_title)

    plt.show()


# plot the stimulus + somatic and dendritic response
# each trial is a different row
def plotResp(data,xmax=1000):
    plt.figure(figsize=(16,8))
    cols = ['#ff9933', '#ff3399', '#3399ff', '#99ff33', '#994c00', '#660033', '#003366', '#336600']
    styles = ['-', '-', '-', '-','-', '-', '-', '-']
    lws = [1, 1, 1, 1, 2, 2, 2, 2]
    # if np.ndim(data.vdata) == 2:
    #     for trace in data.vdata:
    #         plt.plot(data.taxis, trace)
    # else:
    nReps = len(data.vdata)
    nPlots = nReps
    nsyn = data.Ensyn
    nIsyn = data.Insyn
    xmin = 0; xmax=10000

    mat_plots = np.arange(3*nPlots).reshape(3, nPlots)

    for i_rep in np.arange(0,nReps):
        # if (i_rep > 0):
            # xmin = 0
            # xmax = 1100
        i_plot = i_rep 
        ax = plt.subplot(3, nPlots, mat_plots[0,i_plot]+1)
        ind_etimes = (data.stim[:,0] == i_rep + 1)
        if (sum(ind_etimes) > 0) :
            etimes = np.array(data.stim[ind_etimes,:])
            if (len(etimes.shape) == 1) :
                etimes = np.array([data.stim[ind_etimes,:]])
            
            etimes = etimes[:,(1,2)]
            et = etimes[etimes[:,1] < xmax,:]
            if (len(et) > 0):
                ax.plot(et[:,1], et[:,0] + 0.5, '|', color='r')

        ind_itimes = (data.stim[:,0] == -1 * i_rep - 1)
        if (sum(ind_itimes) > 0) :
            itimes = np.array(data.stim[ind_itimes,:])
            if (len(itimes.shape) == 1):
                itimes = np.array([data.stim[ind_itimes,:]])
            itimes = itimes[:,(1,2)]
            it = itimes[itimes[:,1] < xmax,:]
            if (len(it) > 0) :
                ax.plot(it[:,1], it[:,0] + nsyn + 0.5, '|', color='b')

        plt.xlim(xmin,xmax)
        plt.ylim(0,nsyn + nIsyn)


        
        ax = plt.subplot(3, nPlots, mat_plots[1,i_plot]+1)
        trace = data.vdata[i_rep]
        ax.plot(data.taxis, trace)
        plt.xlim(xmin,xmax)
        plt.ylim(-75,-0)

        if data.recordDend:
            trace = data.vDdata[i_rep]
            nDends = len(trace)
            Vlabels = []
            for i_dend in np.arange(0,nDends):
                Dtrace = trace[i_dend]
                if (i_dend < len(data.locDendRec)):
                    Vlabels.append('dend-' + str(data.locDendRec[i_dend]))
                else:
                    Vlabels.append('axon')
                if (i_dend == 0):
                    ax = plt.subplot(3, nPlots, mat_plots[2,i_plot]+1)
                ax.plot(data.taxis, Dtrace, color=cols[i_dend], linestyle=styles[i_dend], lw=lws[i_dend], label=Vlabels[i_dend])
                legend = ax.legend(loc=0, shadow=False, fontsize='small')

            plt.xlim(xmin,xmax)
            plt.ylim(-80,0)
    plt.show()
