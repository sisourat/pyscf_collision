 # input

debug = False
analyze = False
nodiag = True
nstep_analysis = 10
orb =  "HF" #"modpot"  # HF or modpot

ne = 4
tdoc_frozen = 0

tbasis = {'C': 'ccpvdz' }
tgeom = "C 0 0 0.0 ; "
talp = [0.0,7.1,7.1]  # for modpot  V = sum_i c_i * exp(-alp_i*(r-r_i)) * (r-r_i)**n_i tcoef=c_i; talp_i=alp_i, center=r_i, power=n_i
tcoef = [-4.0,-2.0,-7.1]
tcenter = [[0,0,0],[0,0,0],[0,0,0]]
tpower = [-1,-1,0]
tcharge = 2
tspin = 0

pbasis = {'He': 'ccpvdz'}
elp = "He"
xp = 0
yp = 0
zp = -1000.0
palp = [0.0]
pcoef = [-2.0]
ppower = [-1]
pcenter = [[0,0,zp]]
pgeom = elp + " " + str(xp) + " " + str(yp) + " " + str(zp)
pcharge = 0
pspin = 0

i_init = 9
dtime = 0.05

zmax = 60.0
ngrid = 60
gridtype = 'exp'  #lin or exp
vproj = 1.0
bmin =  2.5
bmax =  10.5
nbb = 1
xmlfile = 'csfs_c2+he.xml'
