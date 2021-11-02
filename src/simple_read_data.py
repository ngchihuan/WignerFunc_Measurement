import csv
import ast
import numpy as np

def read_file2(filename):
    """
    Returns:
    r -- raw data, shape of r (M,) where M is the number of scanning parameters.
    """
    with open(filename, 'rt') as file:
        d = []
        h = []
        r = []
        t = []
        for row in csv.reader(file):
            data = []
            extra = []
            #print(row)
            
            for item in row:
                try:
                    data.append(float(item))
                except:
                    extra.append(ast.literal_eval(item))                                        
            d.append(data)
            t.append(extra[0])
            h.append(extra[1])
            r.append(extra[2])

    #return array(map(list, map(None, *d))), h, r, t
    return np.transpose(d),h,r,t

def get_nexp(pr):
    nexp = []
    for item in pr:
        try:
            nexp.append(len(item['a']))
        except KeyError:
            nexp.append(0)
    return nexp


def process_raw(raw):
    out = []
    for line in raw:
        d = dict()
        for key, val in line:
            try:
                d[key].append(val)
            except KeyError:
                d[key] = [val]
        out.append(d)
    return out


def get_x_y(filename):
    
    data, hist, raw, timestamp = read_file2(filename)
    raw = process_raw(raw)
    nexp = get_nexp(raw)
    if len(data) == 20:  # We scan two parameters
        x = data[0]  # x axis
        y1 = data[4]  # counter 1
        y2 = data[6]  # counter 3
    else:  # we scan only one parameter
        x = data[0]  # x axis,
        y1 = data[3]  # counter 1
        y2 = data[5]  # counter 3
    # Counter 'a' mean is [12]
    # Counter 'b' mean is [13]
    # Counter 'c' mean is [14]
    print('nexp ',nexp)
    nexp=nexp[0]
    err1 = np.sqrt(y1 * (1.0 - y1) / nexp)
    err2 = np.sqrt(y2 * (1.0 - y2) / nexp)
    x = x[::-1]
    y1 = y1[::-1]
    err1 = err1[::-1]
    print('shape of x ', np.shape(x))
    print('x ',x)
    print('shape of y1 ',np.shape(y1))
    print('y1 ', y1)
    return (x, y1, err1, y2, err2)
