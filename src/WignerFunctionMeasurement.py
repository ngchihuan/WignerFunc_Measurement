from os import error
import numpy as np
import fit

class DataFormatError(Exception):
    pass

def check_data_format(data):
    '''
    check if the input data satisfies the following requiresments:
        1. it is a dictionary {x: [], y: [], yerr: []}.
        2. The array must have same size
    if the data format is wrong, raise a Type Error
    '''
    conf = {'x': [], 'y' : [], 'yerr' : [] }
    if (check_structure(data,conf)==False):
        raise DataFormatError("Wrong format for the input data")
    else:
        if (np.min(data['y']) < 0 or np.max(data['y'])>1.0):
            raise DataFormatError("y is out of range (0,1)")




def check_structure(struct, conf):

    if isinstance(struct, dict) and isinstance(conf, dict):
        # struct is a dict of types or other dicts
        return all(k in conf and check_structure(struct[k], conf[k]) for k in struct)
    if isinstance(struct, list) and isinstance(conf, list):
        # struct is list in the form [type or dict]
        return all(check_structure(struct[0], c) for c in conf)
    elif isinstance(struct, type):
        # struct is the type of conf
        return isinstance(conf, struct)
    else:
        # struct is neither a dict, nor list, not type
        return False


def parity_calculate(data):
    try:
        check_data_format(data)
        x = data['x']
        y = data['y']
        yerr = data['yerr']
        res = fit.fit_sum_multi_sine_offset_deve(x, y, max_n_fit, weights, Omega_0, gamma, offset=offset, rsb=True,\
                                         gamma_fixed=False, \
                                   customized_bound_population=None)
    except DataFormatError as err:
        print('Error! {0}'.format(err))
    res = fit.fit_sum_multi_sine_offset_deve(x, y, max_n_fit, weights, Omega_0, gamma, offset=offset, rsb=True,\
                                         gamma_fixed=False, \
                                   customized_bound_population=None)

    parity = 0

    return parity

class WignerFunc_Measurement():
    def __init__(self) -> None:
        self.Sbs=[]
        pass

    def set_path(self,fpath):
        self.path = fpath

    def list_all_files(self):
        self.files = []
        pass

    def SBM_gen(self):
        for f in  self.files:
            print (f)
            self.sb = SideBandMeasurement(f) 
            self.SBs.append(self.sb)


class SideBandMeasurement():
    def __init__(self,fname) -> None:
        self.fname = self.set_fname(fname)
        self.xy = None
        self.plot = None
        self.parity = None

    def set_fname(self,fname):
        self.fname = fname

    def extract_xy(self):
        '''
        Extract xy data from the data files 
        '''
        self.xy = None

    def eval_parity(self):
        parity_calculate(self.xy)

    def plotxy(self):
        self.plot = None

if __name__ == '__main__':
    #data = [1,2,3,4]
    conf = {'x': [], 'y' : [], 'yerr' : [] }
    data = {'x': [1,2,3], 'y' : [1, 2, 3], 'yerr' : [1,2,3] }
    #print(check_structure(data,conf) )
    parity_calculate(data)