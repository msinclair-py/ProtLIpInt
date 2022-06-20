import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
import numpy as np
#from multiprocessing import Pool
from ray.util.multiprocessing import Pool
import itertools
import sys
import os
import json
import matplotlib.pyplot as plt
import collections

#####################
## RUNTIME OPTIONS ##
#####################

system = 'T6R2' # name of psf/dcd for loading and file naming
contact_distance = 3.4 # minimum distance for cutoff calculations (A)
smoothing_cutoff = 3 # hysteresis cutoff for smoothing of bound frames
minimum_bound = 3 # min. number of frames bound to be considered bound

# build custom data structure
class LipidContacts(AnalysisBase):
    def __init__(self, protein, lipids, cutoff=3.4, smoothing_cutoff=3, min_bind=3, **kwargs):
        super().__init__(lipids.universe.trajectory, **kwargs)
        self.lipids = lipids
        self.protein = protein
        #self.u = self.protein.universe
        self.u = self.lipids.universe
        #self.lipids = lipids
        self.cutoff = cutoff
        self.smoothing_cutoff = smoothing_cutoff
        self.min_bind = min_bind

        if np.unique(self.protein.segids).shape[0] > 1:
            self.complex = True
        else:
            self.complex = False


    ###################
    # PRIVATE METHODS #
    ###################

    def _prepare(self):
        '''
        Preprocessing: Set up custom data structure based on full
        TM region residues
        '''

        tm = self.identify_tm()
        self.interactions = self.construct_interaction_array(tm)
        self.mapping = self.map_lipids()


    def _single_frame(self):
        '''
        What to do at each frame
        '''

        frame = self._ts.frame

        # iteration through the tm residues, find unique lipid contacts
        for key in self.interactions.keys():
            res, seg = key.split('-')
            #lips = self.lipids.select_atoms(f'segid MEMB and around \
            #        {self.cutoff} global (resid {res} and segid {seg} and \
            #        protein)', updating=True)
            lips = self.lipids.select_atoms(f'segid MEMB and around \
                    {self.cutoff} global (resid {res} and segid {seg} and \
                    group protein)',protein=self.protein)
            lip_resi = lips.residues.ix # this is the list of unique lipid resIDs for contacts
            if len(lip_resi) > 0:
                lip_resn = [self.mapping[resID] for resID in lip_resi]

                for (lipID, lipRN) in zip(lip_resi, lip_resn):
                    if lipID not in self.interactions[key][lipRN].keys():
                        self.interactions[key][lipRN].update({lipID:[frame]})
                    else:
                        self.interactions[key][lipRN][lipID] += [frame]


#    def _conclude(self):
#        '''
#        Postprocessing:
#        '''
#        with open("raw_interactions.json", "w") as f:
#            json.dump(self.to_json(self.interactions), f)
#
#        self.results = {}
#        for pres in self.interactions.keys():
#            for lip in self.interactions[pres].keys():
#                # check for empty
#                if self.interactions[pres][lip]:
#                    coeffs = self.get_coeff(self.interactions[pres][lip])
#                    self.results.update({f'{pres}-{lip}': coeffs})
#        
#        with open("coefficients.json", "w") as f:
#            json.dump(self.to_json(self.results), f)


    ##################
    # PUBLIC METHODS #
    ##################

    def construct_interaction_array(self, tm_residues):
        '''
        Generate a nested dict structure to track lipid contacts on a per
        reside basis
        '''

        lipids_of_interest = ['PC','PE','PG','PI','PS','PA','CL','SM','CHOL']
        inter = {key:{lip:{} for lip in lipids_of_interest} for key in tm_residues}

        return inter


    def identify_tm(self):
        # find phosphate plane for membrane boundary
        memb_zcog = self.lipids.center_of_geometry()[2] # already defined
        z_top = self.u.select_atoms(f'name P and prop z > {memb_zcog}').center_of_geometry()[2]
        z_bot = self.u.select_atoms(f'name P and prop z < {memb_zcog}').center_of_geometry()[2]

        # obtain list of resids pertaining to residues within this boundary
        protein_residues = self.u.select_atoms(f'protein and prop z > {z_bot} and prop z < {z_top}').residues

        return [f'{resID}-{segID}' for (resID,segID) in zip(protein_residues.ix,protein_residues.segids)]


    def map_lipids(self):
        lips = u.select_atoms('segid MEMB and name P').residues
        lip_mapping = {name:name[-2:] if name != 'CHOL' else name for name in lips.resnames}
        mapping = {resid:lip_mapping[resn] for resid, resn in zip(lips.ix, lips.resnames)}

        return mapping


    def get_coeff(self, simdata):
        '''
        Obtain the distribution of binding events in order to fit an exponential.
        Returns the coefficients of said exponential to be used as training/test data.
        '''
        events = []
        for key in simdata.keys():
            events += self.get_binding_profile(simdata[key])
        
        # throw out minimal binding
        culled = list(filter(lambda inp: inp > self.min_bind, events))
        if not culled:
            return 0

        # fit exponential to `culled` distribution
        hist = np.histogram(culled, bins=50, density=True)
        X, Y = ((hist[1][:-1] + hist[1][1:]) / 2), hist[0]
        print(X,Y)

        coeff = np.polyfit(X, Y, 2, w=np.sqrt(Y))

        return coeff


    def get_binding_profile(self, pairdata):
        # history is used to track the local binding history to handle edge cases
        history = [0]*20
        events = []
        lastframe = pairdata[-1]

        i = 0
        while i <= lastframe:
            # check if bound in this frame
            bound = 1 if pairdata[0] == i else 0

            if bound:
                # if you have been bound within the hyst cutoff
                # you are considered `resident`
                resident = 1 if sum(history[:self.smoothing_cutoff]) > 0 else 0
                history.insert(0, 1)
                pairdata.pop(0)

            else:
                resident = 0
                history.insert(0, 0)

                try:
                    events.append(current)
                except Exception as e:
                    pass

            history.pop()


            if bound and resident:
                current += 1
            elif bound and not resident:
                current = 1

            i += 1

        # need to check last frame for binding since this would not be appended otherwise
        if sum(history[:self.smoothing_cutoff]) > 0:
            events.append(current)

        return events


    def key_to_json(self, data):
        if data is None or isinstance(data, (bool, int, str)):
            return data
        if isinstance(data, (tuple, frozenset, np.int64)):
            return str(data)
        raise TypeError


    def to_json(self, data):
        if data is None or isinstance(data, (bool, int, tuple, range, str, list)):
            return data
        if isinstance(data, (set, frozenset)):
            return sorted(data)
        if isinstance(data, dict):
            return {self.key_to_json(key): self.to_json(data[key]) for key in data}
        raise TypeError

#####--------------------------------------------------------------------------------------#####
###----------END OF CLASS------------------------FUNCTIONS GO HERE---------------------------###

def get_coeffs(raw_data):
    coeffs = {}

    for protID in raw_data.keys():
        coeffs.update({protID:{}})
        for restype in raw_data[protID].keys():
            coeffs[protID].update({restype:[]})

            coeff = smoothing(raw_data[protID][restype])
            coeffs[protID][restype] = coeff

    return coeffs


def smoothing(simdata, min_bind = 3):
    '''
    Obtain the distribution of binding events in order to fit an exponential.
    Returns the coefficients of said exponential to be used as training/test data.
    '''
    events = []
    for key in simdata.keys():
        events += get_binding_profile(sorted(simdata[key]))
    
    # throw out minimal binding
    culled = list(filter(lambda inp: inp > min_bind, events))
    if not culled:
        return 0

    # fit exponential to `culled` distribution
    hist = np.histogram(culled, bins=50, density=True)
    X, Y = ((hist[1][:-1] + hist[1][1:]) / 2), hist[0]
    print(X,Y)

    coeff = np.polyfit(X, Y, 2, w=np.sqrt(Y))

    return coeff.tolist()


def get_binding_profile(pairdata, smoothing_cutoff = 3):
    # history is used to track the local binding history to handle edge cases
    history = [0]*20
    events = []
    lastframe = pairdata[-1]

    i = 0
    while i <= lastframe:
        # check if bound in this frame
        bound = 1 if pairdata[0] == i else 0

        if bound:
            # if you have been bound within the hyst cutoff
            # you are considered `resident`
            resident = 1 if sum(history[:smoothing_cutoff]) > 0 else 0
            history.insert(0, 1)
            pairdata.pop(0)

        else:
            resident = 0
            history.insert(0, 0)

            try:
                events.append(current)
            except Exception as e:
                pass

        history.pop()


        if bound and resident:
            current += 1
        elif bound and not resident:
            current = 1

        i += 1

    # need to check last frame for binding since this would not be appended otherwise
    if sum(history[:smoothing_cutoff]) > 0:
        events.append(current)

    return events


def merge_data(nJSONs):
    for i in range(nJSONs):
        with open(f'datafiles/raw_interactions{i}.json', 'r') as infile:
            data = json.load(infile)

        # this means we are appending data to the final data structure
        if not i == 0:
            for protID in final.keys():
                for restype in final[protID].keys():
                    for lipID in final[protID][restype].keys():
                        
                        # this is a try statement since `data` may not contain
                        # this particular lipID
                        try:
                            frames = data[protID][restype][lipID]
                            final[protID][restype][lipID] += frames
                        except KeyError:
                            pass

                    # this is to capture any lipIDs from `data` that we have
                    # yet to see in `final`
                    for lipID in data[protID][restype].keys():
                        if lipID not in final[protID][restype].keys():
                            final[protID][restype].update({lipID:data[protID][restype][lipID]})

        else:
            final = data

    return final


def key_to_json(data):
    if data is None or isinstance(data, (bool, int, str)):
        return data
    if isinstance(data, (tuple, frozenset, np.int64)):
        return str(data)
    raise TypeError


def to_json(data):
    if data is None or isinstance(data, (bool, int, tuple, range, str, list)):
        return data
    if isinstance(data, (set, frozenset)):
        return sorted(data)
    if isinstance(data, dict):
        return {key_to_json(key): to_json(data[key]) for key in data}
    raise TypeError


###-------------------------------------------------###
#---------------CODE STARTS HERE----------------------#
###-------------------------------------------------###

filepath = '/Scr/msincla01/YidC_Membrane_Simulation/Equilibrium_Sims/Replicas/Top6'
dcds = [f'{filepath}/{system}_S{n}.dcd' for n in range(1,10)]

u = mda.Universe(f'{filepath}/{system}.psf',
                 dcds)

protein = u.select_atoms('protein')
lipids = u.select_atoms('segid MEMB')

lipid_analysis = LipidContacts(protein, lipids, cutoff = contact_distance,
                                smoothing_cutoff = smoothing_cutoff, 
                                min_bind = minimum_bound)

def parallelize_run(analysis, n_workers, worker_id):
    analysis.run(start=worker_id, step=n_workers, verbose=not worker_id)
    return analysis

def display_hack():
    sys.stdout.write(' ')
    sys.stdout.flush()

n_workers = os.cpu_count()

params = list(zip(itertools.repeat(lipid_analysis),
             itertools.repeat(n_workers),
             range(n_workers)))


#params = list(zip([lipid_analysis for _ in range(n_workers)],
#             itertools.repeat(n_workers),
#             range(n_workers)))


# This is REQUIRED in order for multiprocessing to work
if __name__ == "__main__":
#	pool = Pool()
#	analyses = pool.map(lipid_analysis,n_workers)
#	pool = Pool(processes=n_workers, initializer=display_hack)
#	analyses = pool.starmap(parallelize_run, params)
#	pool.close()

	import ray
	ray.init()

	@ray.remote
	def parallelize_run(analysis, n_workers, worker_id):
		analysis.run(start=worker_id, step=n_workers, verbose=not worker_id)
		return analysis

	futures = [parallelize_run.remote(par[0],par[1],par[2]) for par in params]
	print(ray.get(futures))

    # dump data into files for checkpointing purposes
	n_frames = [partial_analysis.n_frames for partial_analysis in analyses]
	data = [partial_analysis.interactions for partial_analysis in analyses]
	
	if not os.path.exists('datafiles/'):
		os.mkdir('datafiles/')
		
	print(f'Writing out {n_workers} data files.')
	for i, d in enumerate(data):
		with open(f'datafiles/raw_interactions{i}.json', 'w') as f:
			json.dump(to_json(d), f)

    # combine all data into master checkpoint file, clean up files
	print('Writing out master data file and cleaning up` datafiles/`')
	master = merge_data(n_workers)
	with open(f'datafiles/raw_data_{system}.json', 'w') as f:
		json.dump(to_json(master), f)
		
	for i in range(n_workers):
		os.remove(f'datafiles/raw_interactions{i}.json')

    # smooth and then obtain coefficients for entire dataset
	print('Smoothing frame data, fitting curves and calculating coefficients')
	coeffs = get_coeffs(master)
	with open(f'datafiles/{system}_coeffs.json', 'w') as f:
		json.dump(to_json(coeffs), f)


