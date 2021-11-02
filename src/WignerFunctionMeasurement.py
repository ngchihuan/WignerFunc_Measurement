from os import error
from shutil import Error
import numpy as np
import fit
import simple_read_data

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

def print_debug():
    debug_msg = 'debug'
    return  debug_msg
    


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
    def __init__(self,fname,raw = False ) -> None:
        self.fname = fname
        self.xy = dict((el,[]) for el in ['x','y','yerr'])
        self.plot = None
        self.parity = None
        self.raw = raw

        self.extract_xy()
        self.weight = [1, 0, 0]
        self.Omega_0 = 0.05
        self.gamma = 1e-4
        self.offset = 0.0

    def set_Omega(self,omega):
        try:
            self.Omega_0 = float(omega)
        except ValueError:
            print('Rabi freq must a nummber')
            raise ValueError



    def set_weight(self,weight):
        try:
            self.weight = [float(i) for i in weight]
        except TypeError as err:
            print(err)
            raise TypeError

        except ValueError:
            print('the given weight does not contain float numbers')
            raise ValueError

        return None

    def extract_xy(self):
        '''
        Extract xy data from the data files 
        '''
        if self.raw== True:
            try:
                (self.xy['x'], self.xy['y'], self.xy['yerr'],_,_) = simple_read_data.get_x_y(self.fname)
            except Error as err:
                print('There are some errors ',err)
        else:
            try:
                self.xy['x'], self.xy['y'], self.xy['yerr'] = tuple(np.genfromtxt(self.fname))
            except OSError as err:
                print('file \'%s\' is not found' %(self.fname))
                raise OSError
            except ValueError:
                print('data file has wrong data format')
        
    def extract_pop(self):
        res= None
        try:
            #check_data_format(data)
            x = self.xy['x']
            y = self.xy['y']
            yerr = self.xy['yerr']
            res = fit.fit_sum_multi_sine_offset(x, y, yerr, self.weight, self.Omega_0, self.gamma, offset = self.offset, rsb=False\
                ,gamma_fixed=False,customized_bound_population=None)

        except DataFormatError as err:
            print('Error! {0}'.format(err))
        
        except RuntimeError:
            print('Could not find the optimal fitting parameters')

        except Error as e:
            print(e)

        return res

    def eval_parity(self):
        res = self.extract_pop()
        self.weight_fit = res['weight fit']
        self.parity = 0 
        for i,j in enumerate(self.weight_fit):
            if i%2 == 0:
                self.parity += j*1 
            else:
                self.parity += j*(-1)
        return self.parity
        #use map and filter to do it in a better way???

    def plotxy(self):
        self.plot = None

if __name__ == '__main__':
    pass
    #data = [1,2,3,4]
    #conf = {'x': [], 'y' : [], 'yerr' : [] }
    #data = {'x': [1,2,3], 'y' : [1, 2, 3], 'yerr' : [1,2,3] }
    #print(check_structure(data,conf) )
    #parity_calculate(data)
    fname2 = 'test_data/r1op bsb delay scan after sbc Xcohex 50us phase 0'
    sb1 = SideBandMeasurement(fname2,raw=True)