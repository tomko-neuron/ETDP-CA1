COMMENT
This is a synaptic mechanism implementing a form of stream-specific voltage based event-timing-dependent plasticity (ETDP).
The mechanism uses simple STDP rule that depends entirely on the interval between the pre- and postsynaptic spikes. This is the nearest-neighbor multiplicative implementation of STDP, that is PRESYNAPTIC CENTRED (Morrison et al. 2008):
each presynaptic spike (presynaptic event) is paired with the last postsynaptic event and the next postsynaptic event. 
The postsynaptic event is detected when the voltage at the site of the synapse exceeds a threshold value.

When a postsynaptic event occurs, the mechanism receives an event with flag == 2. The FOR_NETCONS loop iterates over all
NetCons that target this particular instance of the synaptic mechanism. It increases each NetCon's weight by
a multiplicative factor that depends on the latency between the time of the most recent event (presyn. spike) that was
delivered by that NetCon and the time of the postsynaptic event.

When a presynaptic spike occurs, the mechanism receives an event with flag == 0. The weight wE associated with the NetCon
that delivered the spike event is depressed by a multiplicative factor that depends on tpost-t, which is the length of time
that has elapsed since the most recent postsynaptic event.

The mechanism shown here is a functional model of STDP as opposed to a mechanistic model, which would involve some
representation of the biological processes that account for the plasticity. If you need a mechanistic model of ETDP that
involves processes in the presynaptic terminal, then the equations that describe those processes must be explicitly
included in the implementation.

This NMODL code also has a WATCH keyword that can be used to monitor membrane potential at the postsynaptic site for
occurrence of an action potential (helpful for implementing STDP), and a FOR_NETCONS keyword
so that a single instance of an ETDP mechanism can deal properly with multiple input streams.

FOR_NETCONS(same, args, as, netreceive) { stmt } iterates over all NetCon objects that have this POINT_PROCESS
or ARTIFICIAL_CELL as the target. The arglist must have the same number of args as the NET_RECEIVE arg list and must have
the same units. If the units are missing, they are inherited by the corresponding arg of the NET_RECEIVE block.
If an arg has the same name, it hides the NET_RECEIVE arg. The purpose is to allow straightforward specification of certain
kinds of plasticity based on post synaptic state in the context of generalized synapses or artificial cells.

Exp2SynETDP is event target, not event source. NetCons convey the fact that an event happened, plus a weight vector and
a flag variable. The flag variable is set by code in the target mechanism's NET_RECEIVE block.

Two state kinetic scheme synapse described by rise time tau1,
and decay time constant tau2. The normalized peak condunductance is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

If tau2-tau1 -> 0 then we have a alphasynapse.
and if tau1 -> 0 then we have just single exponential decay.

The "factor" is evaluated in the
initial block such that an event of weight 1 generates a
peak conductance of 1.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

ENDCOMMENT

NEURON {
	POINT_PROCESS Exp2SynETDP_multNNb_precentred
	RANGE tau1, tau2, e, i, dtau, ptau
	RANGE wMax 	:hard bound 
	NONSPECIFIC_CURRENT i
	RANGE g, tpost, start, pathway
	RANGE d, p
	
}

DEFINE EpspTimes 10000	:to store presynaptic epsp times into an array

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau1=.1 (ms) <1e-9,1e9>
	tau2 = 10 (ms) <1e-9,1e9>
	e=0	(mV)
	dtau = 36 (ms) 			: depression effectiveness time constant (time it takes to fall from 1->0.37)
	ptau = 26 (ms) 			: Lin et al. (Eur J Neurosci 2006) dtau 36 ms, ptau 26 ms, Bi & Poo (1998, 2001) dtau 34 ms, ptau 17 ms
	wMax = 0.005 (uS)		: hard upper bound 
	start = 30000 (ms)		: synapses are allowed to start changing after start ms
	pathway = -1
}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
	factor
	tpost (ms)
	presyntime[EpspTimes] (ms)	: array to store epsp times
	counter						: to count epsps until first post-AP comes
	total (uS)
	flagOLD
	flag_EPSP
	d							: depression factor (multiplicative to prevent < 0)
	p							: potentiation factor (multiplicative)
}

STATE {
	A (uS)
	B (uS)
}

INITIAL {
	LOCAL tp
	total = 0
	if (tau1/tau2 > .9999) {
		tau1 = .9999*tau2
	}
	A = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor
	tpost = -100000
	:printf("INITIAL: tp=%g factor=%g\n", tp, factor)

	FROM i=0 TO EpspTimes-1 {
		presyntime[i]=-1e9
	}
	counter = 0
	
:flag is an implicit argument to NET_RECEIVE. It is an integer, zero by default but a nonzero value can be specified via the second argument to :net_send(). net_send() is used to launch self-events. A self-event comes back to the mechanism that launched it, no NetCon required. 
	net_send(0, 1)
	flagOLD = 1
	flag_EPSP = 0
}

BREAKPOINT { 
	SOLVE state METHOD cnexp
	g = B - A
	i = g*(v - e)
}

DERIVATIVE state {
	A' = -A/tau1
	B' = -B/tau2
}

:The items in the argument list in the NET_RECEIVE statement of a synaptic mechanism are actually the elements of a
:weight vector.  ExpSyn has only one argument in its NET_RECEIVE statement, so a NetCon nc that targets an ExpSyn
:has a weight vector with length 1, which may be called nc.weight or nc.weight[0]. A NetCon called nc that targets
:an ExpSynETDP or an Exp2SynETDP has a weight vector with length 2.  The first element, which is called nc.weight[0],
:corresponds to the first item in the NET_RECEIVE argument list--w--which remains constant during a simulation.
:The second element is called nc.weight[1], and it corresponds to wE.
:The value of this second element DOES change as a consequence of STDP.

NET_RECEIVE(w (uS), wE (uS), tpre (ms), X) {
	INITIAL { wE = w  tpre = -1e9	 X=0 }
	
	if (flag == 0) {
		:printf("Presynaptic spike: entry flag=%g t=%g wInitial=%g wE=%g X=%g tpre=%g tpost=%g\n", flag, t, w, wE, X, tpre, tpost)
		
		A = A + wE*factor
		B = B + wE*factor
		tpre = t
		counter = counter + 1
		presyntime[counter-1] = tpre
		
		if (t > start) {
			X = d * exp((tpost - t)/dtau)
			wE = wE * (1 - X)
			if (wE > 0) {} else {wE = 0}			
		}
		flagOLD = flag
	}
	else if (flag == 2) {
		:printf("Postsynaptic spike: entry flag=%g, flagOLD=%g, t=%g tpost=%g\n", flag, flagOLD, t, tpost)
		
		FOR_NETCONS(w1, wE1, tpres, X1) {
			if (flagOLD==flag) {
				:printf("flagOLD %g = flag %g\n", flagOLD, flag)
			} 
			else {
				if (t > start) {
					FROM i=0 TO counter-1 {
						X1 = p*exp((presyntime[i] - t)/ptau)
						wE1 = wE1*(1+X1)
					}
					if (wE1<wMax) {} 
					else {
						printf("Max weight")
						wE1=wMax
					}
					FROM i=0 TO counter-1 {
						presyntime[i]=-1e9
					}
					counter = 0	
				}
			
			}
		
		}
		tpost = t 
		flagOLD = flag
	}
	WATCH (v > -37) 2
}

