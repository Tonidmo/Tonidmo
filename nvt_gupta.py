import numpy as np
from ase.data import atomic_numbers, reference_states
from ase.md.andersen import Andersen
import ase.io
from asap3 import *
from asap3 import Atoms, EMT, units, Trajectory
from asap3.md.langevin import Langevin
from asap3.md.velocitydistribution import *
from asap3.mpi import world
from GuptaParameters import AuCu_parameters as AuCu_Gupta
from GuptaParameters import AuPt_parameters as AuPt_Gupta

atoms = ase.io.read('Strut.xyz')

atoms.set_cell(100 * np.identity(3))

calc = RGL(AuPt_Gupta)
atoms.set_calculator(calc)

dyn = Andersen(atoms, timestep=5*units.fs, temperature_K = 600, andersen_prob = 10e-2, fixcm=False)

# Make a trajectory writing output
trajectory = Trajectory("TrajectoryMD-output.traj", "w", atoms)
dyn.attach(trajectory, interval=150)

# Print energies
import time
tick = time.time()
with open('Energy.out', 'w') as outfile:
    outfile.write("{0:10s}  {1:10s}  {2:10s}  {3:10s}  {4:10s}".format(
        'Time (ps)', 'E_Kin (ev)', 'E_Pot (ev)', 'E_Tot (ev)', 'Temp (K)\n'))

#Write an output file of energies
def write_energy(a, step=[0,]):
    ekin = a.get_kinetic_energy()
    epot = a.get_potential_energy()
    with open('Energy.out', 'a') as outfile:
        outfile.write("%10d: %-10.5f  %-10.5f  %-10.5f  %10.1f K\n" %
               (step[0], ekin, epot, ekin+epot, 2.0/3.0*ekin/units.kB/len(a)))
    step[0] += 1

write_energy(atoms)

dyn.attach(write_energy, 200, atoms)
dyn.run(2000)

print ("The output is in the ASE Trajectory file TrajectoryMD-output.traj\n")
print("Simulation of %s ns took %s seconds."%(2000*5*10e-6,time.time()-tick))