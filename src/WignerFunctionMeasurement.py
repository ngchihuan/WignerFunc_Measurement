import os
from os.path import join, isfile
from shutil import Error
from sys import exec_prefix
import numpy as np
import fit
import simple_read_data
from tabulate import tabulate
import logging

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
    def __init__(self,fpath,debug=False) -> None:
        self.sb_list={} #dictionary that stores sb measurement
        
        self.set_path(fpath)
        self.list_all_files()

        self.logger = logging.getLogger(__name__)
        self.debug = debug
        if debug == True:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)


    def set_path(self,fpath) -> None:
        try:
            os.listdir(fpath)
            self.fpath = fpath
        except (NotADirectoryError, FileNotFoundError):
            self.logger.error('The given path is not a directory')
            raise WrongPathFormat
        

    def list_all_files(self):
        print('Scanning the directory')
        self.files = [f for f in os.listdir(self.fpath) if isfile(join(self.fpath, f)) and os.path.splitext(join(self.fpath,f))[1] in ['','.dat'] ]
        self.fullpath_files = sorted( [join(self.fpath,f)  for f in os.listdir(self.fpath) if isfile(join(self.fpath, f)) and os.path.splitext(join(self.fpath,f))[1]=='' ] )
        
        if self.files == []:
            self.logger.warning('The directory is empty')
        else:
            print(f'Discovered {len(self.files)} files in the directory')
        return self.files

    def setup_sbs(self):
        print(f'Validating files')
        cnt=0
        for fname in self.fullpath_files:
            try:
                sbs = SideBandMeasurement(fname,raw = False,debug= self.debug)
                
                self.sb_list[str(cnt)] = sbs
                cnt += 1
            except Exception as err:
                pass
            else:
                sbs.eval_parity()
                
        print(f'Discovered {cnt} valid files with right data format\n')
        

    def get_files(self):
        return self.files

    def print_report(self):
        print('Report summary \n')
        t=[[key,sb.folder, sb.short_fname, sb.parity, sb.err_log] for key,sb in self.sb_list.items()]

        print(tabulate(t, headers=['id', 'folder','filename', 'parity','Errors']))

    def refit(self,id,weights=[],omega=None,gamma=None):
        '''
        Refit a sideband measurement using new weights, omega and gamma.
        '''
        if (id >= len(self.sb_list.keys()) ):
            self.logger.warning('id is out of range')
            return
        else:
            sb_target = self.sb_list[str(id)]
            sb_target.reset_log_err()
            print(f'Refitting Sideband measurement {sb_target.fname}')
            if omega!= None:
                sb_target.set_Omega(omega)
            if len(weights) != 0:
                sb_target.set_weight(weights)
            if gamma!= None:
                sb_target.set_gamma(gamma)
            sb_target.eval_parity()
            

    def show_errors(self):
        pass

class SideBandMeasurement():
    def __init__(self,fname,raw = False, debug = False ) -> None:
        self.fname = fname
        self.xy = dict((el,[]) for el in ['x','y','yerr'])
        self.plot = None
        self.parity = None
        self.raw = raw
        self.weight = [1, 0, 0]
        self.Omega_0 = 0.05
        self.gamma = 7e-4
        self.offset = 0.0

        #logging 
        self.err_log=[]
        self.logger= logging.getLogger(__name__)
        if debug == True:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)

        #extract folder name and fname only
        (self.folder, self.short_fname) = self.fname.split("/")[-2:]

        #verify if the data file is valid
        try:
            np.genfromtxt(self.fname)
        except IOError as err:
            #self.logger.exception('file \'%s\' is not found' %(self.fname) )
            raise

        try: 
            self.extract_xy() 
        except ValueError as err:
            #self.logger.exception(err)
            raise



    def log_err(self,errors):
        self.err_log.append(errors)

    def reset_log_err(self):
        self.err_log=[]

    def set_Omega(self,omega):
        try:
            self.Omega_0 = float(omega)
        except ValueError as error:
            self.logger.error(f'Rabi freq must a nummber {error}')
            raise

    def set_gamma(self,gamma):
        try:
            self.gamma = float(gamma)
        except ValueError as error:
            self.logger.error(f'gamma must a nummber {error}')
            raise


    def set_weight(self,weight) -> None:
        self.logger.debug(f'Set weight when fitting sb {self.fname}')
        try:
            self.weight = [float(i) for i in weight]
        except (TypeError,ValueError) as err:
            self.logger.error(f'Set weight error')
            raise 


    def extract_xy(self):
        '''
        Extract xy data from the data files 
        '''
        if self.raw== True:
            try:
                (self.xy['x'], self.xy['y'], self.xy['yerr'],_,_) = simple_read_data.get_x_y(self.fname)
            except Exception as err:
                raise
        else:
            try:
                self.xy['x'], self.xy['y'], self.xy['yerr'] = tuple(np.genfromtxt(self.fname))
            except ValueError as err:
                raise ValueError(f'{self.short_fname} has wrong data format')
        
    def extract_pop(self):

        try:
            self.fit_res = fit.fit_sum_multi_sine_offset(self.xy['x'], self.xy['y'], self.xy['yerr'], self.weight, self.Omega_0, self.gamma, offset = self.offset, rsb=False\
        ,gamma_fixed=False,customized_bound_population=None,debug=False)
        except FloatingPointError as err:
            #self.logger.warning('There is a measurement with zero uncertainty')
            self.log_err('zero sigma')
        
        except Exception as err:
            self.log_err('unexpected error in fitting')
            #raise RuntimeError('Could not fit')

        else:
            redchi = self.fit_res['reduced_chi square']
            if (redchi>10 or redchi<0):
                #self.logger.warning(f'Could not fit well')
                self.log_err(f'Could not fit well, redchi = {round(redchi,2)}')
            return self.fit_res

        
        

    def eval_parity(self):
        self.logger.debug(f'Evaluate parity of {self.fname}')

        res = self.extract_pop()
        if res!= None:
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
    fpath ='../tests/test_data'
    wfm1 = WignerFunc_Measurement(fpath)
    wfm1.setup_sbs()
    wfm1.report()