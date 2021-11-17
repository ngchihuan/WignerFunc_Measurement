import os
from os.path import join, isfile
from sys import exec_prefix
import numpy as np
import fit
import simple_read_data

np.seterr(all='raise')

class DataFormatError(Exception):
    pass

class WrongPathFormat(Exception):
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
    def __init__(self,fpath) -> None:
        self.sb_list={} #dictionary that stores sb measurement
     
        self.set_path(fpath)
        self.list_all_files()


    def set_path(self,fpath) -> None:
        try:
            os.listdir(fpath)
            self.fpath = fpath
        except (NotADirectoryError, FileNotFoundError):
            print('the given path is not a directory')
            raise WrongPathFormat
        

    def list_all_files(self):
        self.files = [f for f in os.listdir(self.fpath) if isfile(join(self.fpath, f)) and os.path.splitext(join(self.fpath,f))[1] in ['','.dat'] ]
        self.fullpath_files = sorted( [join(self.fpath,f)  for f in os.listdir(self.fpath) if isfile(join(self.fpath, f)) and os.path.splitext(join(self.fpath,f))[1]=='' ] )
        if self.files == []:
            print('the directory is empty')
        return self.files

    def setup_sbs(self):
        for id, fname in enumerate(self.fullpath_files):
            sbs = SideBandMeasurement(fname,raw = False)
            sbs.eval_parity()
            self.sb_list[str(id)] = sbs
        

    def get_files(self):
        return self.files

    def report(self):
        print('Report summary \n')
        for id,sb in self.sb_list.items():
            print(id,sb.fname, sb.parity)



class SideBandMeasurement():
    def __init__(self,fname,raw = False ) -> None:
        self.fname = fname
        self.check_file_exist()
        self.xy = dict((el,[]) for el in ['x','y','yerr'])
        self.plot = None
        self.parity = None
        self.raw = raw
        self.weight = [1, 0, 0]
        self.Omega_0 = 0.05
        self.gamma = 7e-4
        self.offset = 0.0
        self.err_log=[]

    def log_err(self,errors):
        self.err_log.append(errors)

    def set_Omega(self,omega):
        try:
            self.Omega_0 = float(omega)
        except ValueError:
            print('Rabi freq must a nummber')
            raise ValueError

    def set_gamma(self,gamma):
        try:
            self.gamma = float(gamma)
        except ValueError:
            print('gamma must a nummber')
            raise ValueError        


    def set_weight(self,weight) -> None:
        try:
            self.weight = [float(i) for i in weight]
        except TypeError as err:
            print(err)
            self.log_err(err)
            raise TypeError

        except ValueError as err:
            print('the given weight does not contain float numbers')
            self.log_err(err)
            raise ValueError

        return None

    def check_file_exist(self):
        try:
            np.genfromtxt(self.fname)
        except OSError as err:
            print('file \'%s\' is not found' %(self.fname))
            raise OSError

    def extract_xy(self):
        '''
        Extract xy data from the data files 
        '''
        if self.raw== True:
            try:
                (self.xy['x'], self.xy['y'], self.xy['yerr'],_,_) = simple_read_data.get_x_y(self.fname)
            except Exception as err:
                print('There are some errors ',err)
                self.log_err.append(err)
        else:
            try:
                self.xy['x'], self.xy['y'], self.xy['yerr'] = tuple(np.genfromtxt(self.fname))

            except OSError as err:
                print('file \'%s\' is not found' %(self.fname))
                raise OSError

            except ValueError as err:
                print('data file has wrong data format')
                raise ValueError
        
    def extract_pop(self):

        try:
            if not self.xy['x']:
                self.extract_xy()
            #check_data_format(data)
            x = self.xy['x']
            y = self.xy['y']
            yerr = self.xy['yerr']
            self.fit_res = fit.fit_sum_multi_sine_offset(x, y, yerr, self.weight, self.Omega_0, self.gamma, offset = self.offset, rsb=False\
                ,gamma_fixed=False,customized_bound_population=None)
            return self.fit_res
        except Exception as err:
            print(err)
            raise err

        

    def eval_parity(self):
        try:
            res = self.extract_pop()
            self.weight_fit = res['weight fit']
            self.parity = 0 
            for i,j in enumerate(self.weight_fit):
                if i%2 == 0:
                    self.parity += j*1 
                else:
                    self.parity += j*(-1)
            return self.parity
        except Exception as err:
            print(err)
            self.log_err(err)
        #use map and filter to do it in a better way???

    def plotxy(self):
        self.plot = None


if __name__ == '__main__':
    fpath ='../tests/test_data'
    wfm1 = WignerFunc_Measurement(fpath)
    wfm1.setup_sbs()
    wfm1.report()