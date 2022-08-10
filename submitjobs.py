#!/usr/bin/env python
import glob, os, re, subprocess

numDCDs = 3
destination = '/Scr/msincla01/Protein_Lipid_DL'
qsub = ['/Scr/msincla01/github/ProtLIpInt/submit.sh']

dirs = [
#     '/Scr/arango/ClC.ec1', 
     '/Scr/arango/membrane_AA/CHL_AB/CHL50_GAMD/chl50.*',
     '/Scr/arango/membrane_AA/abeta/AB.AA91100*',
     '/Scr/arango/nsf-Project/rot.x0.y0/x0.y0.charmm-gui-*/namd-D2',
     '/Scr/sepehr/prestin/replica_1',
     '/Scr/sepehr/prestin/replica_2',
     '/Scr/sepehr/spns/bacterial/6e9c/[1-4]*'
]

# EACH LIST CONTAINS:
#   regex pattern for dcds
#   traj extension (dcd, xtc)
#   numDCDs
#   ncpu
#   traj file prefix (requires divergent path if present)
#   structure file prefix (requires divergent path if present)
#   simulation engine/type

aux = [
#    ['CLC.stride\\.[0-9][0-9].dcd', 'dcd', 
#        numDCDs, '48', 'CLC.stride.', 'CLC_correct', 'namd'],
    ['1\\.[0-9][0-9].dcd', 'dcd', numDCDs, '48', '1.', '1', 'namd'],
    ['1\\.[0-9][0-9].dcd', 'dcd', numDCDs, '48', '1.', '1', 'namd'],
    ['D2.1\\.[0-9].dcd', 'dcd', numDCDs, 
        '48', 'D2.1.', 'hmmm-step5_assembly.xplor_ext', 'hmmm'],
    ['equ\\.[0-9]{1,2}.dcd', 'dcd', numDCDs, '48', 'equ_4/equ.', 'ini/ionized', 'namd'],
    ['equ\\.[0-9]{1,2}.dcd', 'dcd', numDCDs, '48', 'equ_4/equ.', 'ini/ionized', 'namd'],
    ['equ\\.[0-9].dcd', 'dcd', numDCDs, '48', 'equ_3/equ.', 'ini/ionized', 'namd']
]

def natural_sort(l: list) -> list:
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def glob_re(pattern: str, strings: list) -> list:
    fpath = os.path.commonpath(strings)
    if len(strings) == 1:
        fpath = '/'.join(fpath.split('/')[:-1])

    temp = list(map(lambda x: os.path.basename(x), strings))

    culled = filter(re.compile(pattern).match, temp)
    return [ os.path.join(fpath, x) for x in culled ]


def get_jname(p1: str, p2: str) -> str:
    temp = os.path.relpath(p1, p2).split('../')[-1]
    if '/' in temp:
        return temp.split('/')[0]
    return temp


def get_paths(p1: str, p2: str) -> str:
    r1 = '/'.join(p1.split('/')[:-1])
    r2 = '/'.join(p2.split('/')[:-1])
    
    cpath = os.path.commonpath([r1,r2])
    path1 = p1.split(cpath)[-1][1:]
    path2 = p2.split(cpath)[-1][1:]
    return cpath, path1, path2


def append_multi_directory(dirs: list, pat: str, ext: str, 
                            numT: int, nproc: str, traj: str,
                            psf: str, sim: str) -> dict:
    
    tempdict = dict()
    if len(dirs) > 1:
        jnames = [get_jname(dirs[0],dirs[1])]
        for i in range(1,len(dirs)):
            jnames.append(get_jname(dirs[i],dirs[0]))
    else:
        jnames = [os.path.basename(dirs[0])]

    for i, directory in enumerate(dirs):
        trajlist = glob_re(pat, glob.glob(f'{directory}/{traj}*{ext}'))
        trajlist = natural_sort(trajlist)
        numtrajs = len(trajlist)
        numreplicas = numtrajs//numT
        
        # add common path and process any divergence in pathing to
        # traj vs structure file
        cpath, traj, strc = get_paths(f'{directory}/{traj}', 
                                        f'{directory}/{psf}')

        info = [jnames[i],cpath,traj]

        values = [info+[f'{a*numT+1}',
                        f'{a*numT+numT}',
                        f'{strc}',
                        sim] for a in range(numreplicas)]

        if not values:
            values = [info+['1',f'{len(trajlist)}',
                            strc,sim]]


        tempdict.update({directory:values})
    return tempdict


submissions = dict()
for i, _dir in enumerate(dirs):
    if '*' in _dir:
        temp = natural_sort(glob.glob(_dir))
    else:
        temp = [ _dir ]

    submissions.update(append_multi_directory(temp, aux[i][0], aux[i][1],
                                        aux[i][2], aux[i][3], aux[i][4],
                                        aux[i][5], aux[i][6]))

os.chdir(destination)

with open('submitter.txt','w') as outfile:
    for _dir in submissions.keys():
        if isinstance(submissions[_dir][0], list):
            for submission in submissions[_dir]:
                submit = qsub + submission
                outfile.write(' '.join(submit))
                #print(f'Submitting: {submit}.....')
                #subprocess.run(submit)
        else:
            print('ERR OAR!')

    outfile.close()
