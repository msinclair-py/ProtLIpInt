#!/usr/bin/env python
import glob, os, subprocess

numDCDs = 3

def natural_sort(l: list) -> list:
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def glob_re(pattern: str, strings: list) -> list:
    fpath = os.path.commonpath(strings)
    temp = list(map(lambda x: os.path.basename(x), strings))
    culled = filter(re.compile(pattern).match, temp)
    return [os.path.join(fpath, x) for x in culled]

def append_multi_directory(dirs: list, pat: str, ext: str, 
                            numT: int, info: list) -> dict:
    for directory in dirs:
        trajlist = glob_re(pat, glob.glob(f'{directory}/*{ext}'))
        trajlist = natural_sort(trajlist)
        numtrajs = len(trajlist)
        numreplicas = numtrajs//numT

        values = [info+[f'{a}',f'{a+numDCDs-1}'] for a in range(numreplicas)]
        return {directory:values}}

andres = dict('/Scr/arango/ClC.ec1/': ['CLC_correct','32','CLC.stride.','1','1'])

#'/Scr/arango/membrane_AA/abeta/AB.AA91100'

temp = natural_sort(glob.glob('/Scr/arango/membrane_AA/CHL_AB/CHL50_GAMD/chl50.*'))
for _dir in temp:
    andres.update(append_multi_directory(_dir, '1\\\\.[0-9][0-9]\\\\.dcd', 
                                        'dcd', numDCDs, ['CHL_AB','32','1.']))


print(andres)

for _dir in andres.keys():
    if isinstance(andres[_dir][0], list()):
        sd
    elif isinstance(andres[_dir][0], str):
        sdfadf
    else:
        print('ERR OAR!')
