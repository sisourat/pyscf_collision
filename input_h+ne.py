 # input
debug = False
analyze = False
nodiag = True
nstep_analysis = 10

orb =  "HF"  # HF or modpot

ne = 10
tdoc_frozen = 0

tbasis = {'Ne': 'ccpvdz' }
tgeom = "Ne 0 0 0.0 ; "
#talp = [0.0,2.422,2.422]  # for modpot  V = sum_i c_i * exp(-alp_i*(r-r_i)) * (r-r_i)**n_i tcoef=c_i; talp_i=alp_i, center=r_i, power=n_i
#tcoef = [-5.0,-4.0,-0.461]
#tcenter = [[0,0,0],[0,0,0],[0,0,0]]
#tpower = [-1,-1,0]
tcharge = 0
tspin = 0

pbasis = {'H': 'ccpvdz'}
elp = "H"
xp = 0
yp = 0
zp = -1000.0
#palp = [0.0]
#pcoef = [-1.0]
#ppower = [-1]
pcenter = [[0,0,zp]]
pgeom = elp + " " + str(xp) + " " + str(yp) + " " + str(zp)
pcharge = 1
pspin = 0

i_init = 5
dtime = 0.05

zmax = 6000.0
ngrid = 60
gridtype = 'exp'  #lin or exp
vproj = 0.4
bmin =  0.5
bmax =  10.5
nbb = 40
xmlfile = 'csfs_h+ne.xml'

