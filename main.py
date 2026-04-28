from pyscf import gto
import numpy as np
import sys
import os
import pathlib
import matplotlib.pyplot as plt
import importlib

from libcollision import *
from libdyn import *
from libanalysis import *
from generate_csfs import *
from cimat import *
from scipy.interpolate import interp1d


if __name__ == "__main__":

  pdir = pathlib.Path().resolve()
  sys.path.append(pdir)
  module_name = sys.argv[1]
  module = importlib.import_module(module_name)
  # Copy all names from the module into the global namespace
  globals().update(vars(module))
  sgeom = tgeom + pgeom
  sbasis = tbasis | pbasis
  scharge = tcharge + pcharge
  sspin = tspin + pspin

  tmol = gto.M(atom=tgeom,basis=tbasis,charge=tcharge,spin=tspin,symmetry = True)
  tmol.build( unit = 'Bohr')
  pmol = gto.M(atom=pgeom,basis=pbasis,charge=pcharge,spin=pspin,symmetry = True)
  pmol.build( unit = 'Bohr')
  mol = gto.M(atom=sgeom,basis=sbasis,charge=scharge,spin=sspin)
  mol.build( unit = 'Bohr')

# Computes target and projectile HF orbitals and ""model"" orbitals
  print("TARGET")
  if orb == 'modpot':
   tmo, tmo_e = modpot_system(tmol,talp,tcoef,tcenter,tpower,debug)
  elif orb == 'HF':
   tmo, tmo_e = system(tmol,debug)
  else:
      raise NotImplementedError("Only HF or modpot orbitals implemented")
  ntmo = len(tmo)

  #print()
  #print(tmo.T)
  #print()

  print("PROJECTILE")
  if orb == 'modpot':
   pmo, pmo_e = modpot_system(pmol,palp,pcoef,pcenter,ppower,debug)
  elif orb == 'HF':
   pmo, pmo_e = system(pmol,debug)
  else:
      raise NotImplementedError("Only HF or modpot orbitals implemented")
  npmo = len(pmo)
  xp = 0
  yp = 0
  zp = -zmax
  pgeom = elp + " " + str(xp) + " " + str(yp) + " " + str(zp)
  pmol = gto.M(atom=pgeom,basis=pbasis,charge=pcharge,spin=pspin)
  pmol.build( unit = 'Bohr')

# Collision dynamics
  blist = np.linspace(bmin,bmax,nbb)
  if(gridtype=='exp'):
    zlist =  np.sort(np.concatenate((-np.logspace(-18, 0, base=2, num=ngrid)*zmax,np.logspace(-18, 0, base=2, num=ngrid)*zmax)))
  elif(gridtype=='lin'):
    zlist =  np.linspace(-zmax,zmax,ngrid)
  else:
    raise NotImplementedError('Exp or Lin Grid only')

  nmo = ntmo + npmo

  h1emo = np.zeros((nmo,nmo))*1.0j
  ovmo = np.zeros((nmo,nmo))*1.0j

  smo = np.zeros((nmo,nmo))
  smo[0:ntmo,0:ntmo] = tmo
  smo[ntmo:nmo,ntmo:nmo] = pmo

  # Checks the asymptotic energies
  xp = 0
  yp = 0
  zp = -zmax
  pgeom = elp + " " + str(xp) + " " + str(yp) + " " + str(zp)
  sgeom = tgeom + pgeom
  mol = gto.M(atom=sgeom,basis=sbasis,charge=scharge,spin=sspin)
  mol.build( unit = 'Bohr')

  zpcenter = [np.repeat([0,0,zp], len(pcenter))]
  if orb == 'modpot':
   salp = np.concatenate((talp, palp), axis=0)
   scoef = np.concatenate((tcoef, pcoef), axis=0)
   scenter = np.concatenate((tcenter, zpcenter), axis=0)
   spower = np.concatenate((tpower, ppower), axis=0)

  csfs = process_xml_csf(xmlfile)
  ncsfs = len(csfs)

  for i in range(ncsfs):
      print(i,csfs[i])

  nep_csf = []
  for csf in csfs:
      _, alpe, betae = csf.terms[0]
      nte = int(np.count_nonzero(np.array(alpe)<ntmo) + np.count_nonzero(np.array(betae)<ntmo))
      npe = len(alpe)+len(betae)-nte
      nep_csf.append(npe)

  # Asymptotic Energies
  phase = np.ones(ncsfs)
  if orb == 'modpot':
   ovl, kin, pot = hcore_modpot(salp, scoef, scenter, spower, mol, smo)
  elif orb == 'HF':
   ovl, kin, pot = hcore(mol, smo)
  else:
      raise NotImplementedError("Only HF or modpot orbitals implemented")
  eri = twoeints(mol,smo)
  if not tdoc_frozen == 0:
      raise NotImplementedError("Frozen core orbitals have to be tested")
  eecore = 2.0*np.trace(eri[0:tdoc_frozen,0:tdoc_frozen,:,:])
  h1e = kin + pot + eecore
  hmat, smat = cimat(ne,nmo,ovl,h1e,eri,csfs,phase)


  # Create a mask for the diagonal
  if nodiag:
   diagonal_mask = np.eye(hmat.shape[0], dtype=bool)
   # Apply the mask to keep only diagonal elements
   diagonal_hmat = hmat * diagonal_mask
   hmat = diagonal_hmat

  eig, eigv = np.linalg.eig(hmat)
  idx = eig.argsort()[::-1]
  eig = eig[idx]
  eigv = eigv[:,idx]
  nsta = len(eig)

  esta = []
  net = []
  nep = []
  print()
  print("Asymptotic energies")
  print()

  # Get the absolute values of the eigenvectors
  abs_eigv = np.abs(eigv)
  # Find the index of the largest component for each eigenvector (column)
  largest_component_indices = np.argmax(abs_eigv, axis=0)
  # Find the values of the largest components
  largest_component_values = abs_eigv[largest_component_indices, range(eigv.shape[1])]


  for i in range(nsta):
    _, alpe, betae =  csfs[largest_component_indices[i]].terms[0]
    esta.append(eig[i])
    print(i, esta[i].real)
    for j in range(ncsfs):
     if(np.abs(eigv[j,i].real)>0.3):
      print("      ",j, eigv[j,i].real, csfs[j])
    print()

  if(len(sys.argv)>2 and sys.argv[2]=='0'):
      sys.exit()

  for b in blist:

   tmat = []
   for zproj in zlist:

     time = zproj/vproj
     phase = []
     for i, csf in enumerate(csfs):
       if(nep_csf[i]==0):
         phase.append(1.0)
       elif(nep_csf[i]==1):
         phase.append(np.exp(-vproj*zproj*1.0j)*np.exp(+0.5*vproj**2*time*1.0j))
       elif(nep_csf[i]==2):
         phase.append(np.exp(-vproj*zproj*1.0j)**2*np.exp(+vproj**2*time*1.0j))
       else:
         raise NotImplementedError("Wrong number of projectile electrons")

     xp = b
     yp = 0
     zp = zproj
     pgeom = elp + " " + str(xp) + " " + str(yp) + " " + str(zp)
     sgeom = tgeom + pgeom
     mol = gto.M(atom=sgeom,basis=sbasis,charge=scharge,spin=sspin)
     mol.build( unit = 'Bohr')

     zpcenter = [np.repeat([xp,yp,zp], len(pcenter))]
     if orb == 'modpot':
      salp = np.concatenate((talp, palp), axis=0)
      scoef = np.concatenate((tcoef, pcoef), axis=0)
      scenter = np.concatenate((tcenter, zpcenter), axis=0)
      spower = np.concatenate((tpower, ppower), axis=0)

     if orb == 'modpot':
       ovl, kin, pot = hcore_modpot(salp, scoef, scenter, spower, mol, smo)
     elif orb == 'HF':
       ovl, kin, pot = hcore(mol, smo)
     else:
       raise NotImplementedError("Only HF or modpot orbitals implemented")
     eri = twoeints(mol,smo)
     if not tdoc_frozen == 0:
       raise NotImplementedError("Frozen core orbitals have to be tested")
     eecore = 2.0*np.trace(eri[0:tdoc_frozen,0:tdoc_frozen,:,:])
     h1e = kin + pot + eecore

     matH, matS = cimat(ne,nmo,ovl,h1e,eri,csfs,phase)

     hmat = np.linalg.inv(eigv) @ matH @ eigv
     smat = np.linalg.inv(eigv) @ matS @ eigv
     inv = np.linalg.inv(smat)
     mat = np.matmul(inv, hmat)
     tmat.append(mat)

   # running dynamics for bproj = b
   tlist = zlist/vproj
   hmat_interp = interp1d(tlist, tmat , axis=0)
   psi0 = np.zeros(ncsfs, dtype=complex)
   psi0[i_init] = 1.0
   ntime = int(2.0*zmax/(vproj*dtime))
   t_grid = np.linspace(zlist[0]/vproj,zlist[-1]/vproj,ntime)
   #wf_t = solve_tdse(hmat_interp, psi0, t_grid)
   wf_t = solve_tdse_sequential(hmat_interp, psi0, t_grid)
   prob = np.abs(wf_t[-1,:])**2

   #amp = wf_t[-1,:]
   #formatted_string = "  ".join([f"{num.real} {num.imag}" for num in amp])
   #print(b, formatted_string)

   formatted_string = "  ".join([f"{num:.6f}" for num in prob])
   print(b, formatted_string,' ', f"{np.sum(prob):.6f}")

   #analyzing the dynamics for bproj = b
   if analyze == True:
    fout = open('prob_time_'+str(vproj)+'_'+str(b)+'.txt','w')
    for it, time in enumerate(t_grid):
       zproj = vproj * time
       if(it%nstep_analysis==0):
        amp2 = np.abs(wf_t[it])**2
        print(zproj,' '.join(map(str,amp2)),np.sum(amp2),file=fout)
    fout.close()
    # can do 1rdm analysis here

    ztime = []
    stime = []
    for it, time in enumerate(t_grid):
       zproj = vproj * time
       if(it%nstep_analysis==0):

        ztime.append(zproj)
        xp = b
        yp = 0
        zp = zproj
        pgeom = elp + " " + str(xp) + " " + str(yp) + " " + str(zp)
        sgeom = tgeom + pgeom
        mol = gto.M(atom=sgeom,basis=sbasis,charge=scharge,spin=sspin)
        mol.build( unit = 'Bohr')

        if orb == 'modpot':
          ovl, kin, pot = hcore_modpot(salp, scoef, scenter, spower, mol, smo)
        elif orb == 'HF':
          ovl, kin, pot = hcore(mol, smo)
        else:
          raise NotImplementedError("Only HF or modpot orbitals implemented")

        cicoeffs = wf_t[it]
        csfs_coeffs = np.zeros(ncsfs,dtype=complex)
        for i, c in enumerate(cicoeffs):
            for ii, cc in enumerate(eigv[:,i]):
                #print(i,c,cc)
                csfs_coeffs[ii]+= c*cc

        #rdm1, rdm2 =  one_rdm_nonorth(csfs, cicoeffs, ovl)
        rdm1 =  one_rdm_nonorth(csfs, csfs_coeffs, ovl)
        rrdm1 = np.real(rdm1)
        #rrdm2 = np.real(rdm2)

        eigenvalues, eigenvectors = np.linalg.eigh(rdm1)
        # Compute von Neumann entropy
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-12))  # Add small value to avoid log(0)
        #print(eigenvalues)
        print(zproj,entropy,*np.diag(rrdm1))
        #import time
        #time.sleep(2.5)
        #stime.append(entropy)
        #print(zproj,rrdm1[0,0],rrdm1[1,1],rrdm1[3,3],rrdm1[4,4],rrdm1[2,2],rrdm1[5,5])
        #print(np.real(rdm1))
        #print(np.trace(np.real(rdm1)))
        #print()

        #print("Diagonal elements of the 2-RDM (pair populations):")
        #for p in range(rdm2.shape[0]):
        # for q in range(rdm2.shape[1]):
        #  print(f"Γ({p},{q},{p},{q}) = {rdm2[p, q, p, q]:.3f}")
        # Sum over p for each q
        #sum_over_p = np.zeros(2*nmo)  # Initialize for each q
        #for q in range(2*nmo):
        #    for p in range(0,2):
        #        sum_over_p[q] += rdm2[p, q, p, q]

        #print(zproj,sum_over_p[2],sum_over_p[5])
        #print(zproj,rrdm2[3,2,3,2],rrdm2[4,2,4,2],rrdm1[3,3],rrdm1[4,4],rrdm1[2,2])
        #print(zproj,' '.join(map(str,np.abs(rdm2[0,2,4,:]))))


    #plt.plot(ztime, stime)
    #plt.show()

