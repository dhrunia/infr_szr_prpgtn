from tvb.simulator.lab import *
import os.path
from matplotlib import colors, cm
import matplotlib.pyplot as pyplt
import time
import scipy.signal as sig
import numpy as np
from InitialConditions import get_equilibrium

def forwardSim(ez,pz,noise,simLen):
  project_dir = "/home/anirudh/Academia/Projects/VEP/data/CJ"
  con = connectivity.Connectivity.from_file(os.path.join(project_dir, "connectivity.zip"))
  con.speed = np.inf
  # normalize
  con.weights = con.weights/np.max(con.weights)
  num_regions = len(con.region_labels)


  pyplt.figure()
  image = con.weights
  norm = colors.LogNorm(1e-7, image.max()) #, clip='True')
  pyplt.imshow(image, norm=norm, cmap=cm.jet)
  pyplt.colorbar()
  #max(con.weights[con.weights != 0])


  epileptors = models.Epileptor(variables_of_interest=['x1', 'y1', 'z', 'x2', 'y2', 'g', 'x2 - x1'])
  epileptors.r = 0.0001
  epileptors.Ks = np.ones(num_regions)*(-1.0)*20.0


  # Patient specific modifications
  #ez = [9]
  #pz = [6, 27]

  epileptors.x0 = np.ones(num_regions)*-2.3
  epileptors.x0[ez] = -1.8
  #epileptors.x0[pz] = -2.05


  coupl = coupling.Difference(a=1.)


  hiss = noise.Additive(nsig = np.array([0.01, 0.01, 0., 0.00015, 0.00015, 0.]))
  if(noise):
    heunint = integrators.HeunStochastic(dt=0.04, noise=hiss)
  else:
    heunint = integrators.HeunDeterministic(dt=0.04)



  #mon_raw = monitors.Raw()
  mon_tavg = monitors.TemporalAverage(period=1.0)
  mon_SEEG = monitors.iEEG.from_file(sensors_fname=os.path.join(project_dir, "seeg.txt"),
                                     projection_fname=os.path.join(project_dir, "gain_inv-square.txt"),
                                     period=1.0,
                                     variables_of_interest=[6]
                                     )
  num_contacts = mon_SEEG.sensors.labels.size


  con.cortical[:] = True     # To avoid adding analytical gain matrix for subcortical sources

  # Find a fixed point to initialize the epileptor in a stable state
  epileptor_equil = models.Epileptor()
  epileptor_equil.x0 = -2.3
  #init_cond = np.array([0, -5, 3, 0, 0, 0])
  init_cond = get_equilibrium(epileptor_equil, np.array([0.0, 0.0, 3.0, -1.0, 1.0, 0.0]))
  init_cond_reshaped = np.repeat(init_cond, num_regions).reshape((1, len(init_cond), num_regions, 1))
  sim = simulator.Simulator(model=epileptors,
                            initial_conditions=init_cond_reshaped,
                            connectivity=con,
                            coupling=coupl,
                            conduction_speed=np.inf,                          
                            integrator=heunint,
                            monitors=[mon_tavg, mon_SEEG])

  sim.configure()


  (ttavg, tavg), (tseeg, seeg) = sim.run(simulation_length=simLen)


#  # Normalize the time series to have nice plots
#  tavgn = tavg/(np.max(tavg, 0) - np.min(tavg, 0))
#  seegn = seeg/(np.max(seeg, 0) - np.min(seeg, 0))
#  seegn = seegn - np.mean(seegn, 0)


  b, a = sig.butter(2, 0.1, btype='highpass', output='ba')
  #seegf = sig.filtfilt(B, A, seegn)
  seegf = np.zeros(seegn.shape)
  for i in range(num_contacts):
      seegf[:, 0, i, 0] = sig.filtfilt(b, a, seeg[:, 0, i, 0])

  # Save data for data fitting
  np.save('../../results/ForwardSim/CJ/complex.npy',np.transpose(seeg[:,0,:,0]))
  np.savetxt('../../results/ForwardSim/CJ/weights.txt',con.weights)
  np.savetxt('../../results/ForwardSim/CJ/centers.txt',con.centres)

#  #Plot raw time series
#  pyplt.figure(figsize=(9,10))
#
#  indf = 0
#  indt = -1
#
#  regf = 0
#  regt = 84
#
#  pyplt.plot(ttavg[indf:indt], tavg[indf:indt, 6, regf:regt, 0]/4 + np.r_[regf:regt], 'r')
#  pyplt.yticks(np.r_[regf:regt], con.region_labels[regf:regt])
#  pyplt.title("Epileptors time series")
#  pyplt.tight_layout()
#  pyplt.show(block=False)   
#
#
#  pyplt.figure(figsize=(10,20))
#  pyplt.plot(tseeg[:], seegn[:, 0, :, 0] + np.r_[:num_contacts])
#  pyplt.yticks(np.r_[:num_contacts], mon_SEEG.sensors.labels[:])
#  pyplt.title("SEEG")
#  pyplt.tight_layout()
#  pyplt.show(block=False)
#
#  pyplt.figure(figsize=(10,20))
#  pyplt.plot(tseeg[:], (seegn[:, 0, 1:num_contacts, 0] - seegn[:, 0, 0:num_contacts-1, 0]) + np.r_[:num_contacts-1])
#  pyplt.yticks(np.r_[:num_contacts], mon_SEEG.sensors.labels[:])
#  pyplt.title("SEEG")
#  pyplt.tight_layout()
#  pyplt.show(block=False)
#
#  pyplt.figure(figsize=(10, 6))
#
#  electrodes = [("FCA'", 7), ("GL'", 7), ("CU'", 6), ("PP'", 1),
#                ("PI'", 5), ("GC'", 8), ("PFG'", 10),
#                ("OT'", 5), ("GPH'", 6), ("PFG", 10)]
#
#
#  for i, (el, num) in enumerate(electrodes):
#      ind = np.where(mon_SEEG.sensors.labels == el + str(num))[0][0]
#      pyplt.plot(tseeg[:], (seegn[:, 0, ind, 0] - seegn[:, 0, ind - 1, 0])/0.5 + i)
#
#  labels = [el[0] + str(el[1]) + "-" + str(el[1] - 1) for el in electrodes]
#  pyplt.yticks(np.r_[:len(electrodes)], labels)
#  pyplt.tight_layout()
#  pyplt.show()


for i in arange(1,5):
  ez = np.randomn.randint(1,85)
  pz = []
  simLen = 10*1000
  noise = False
  forwardSim(ez,pz,noise,simLen)

for i in arange(1,5):
  ez = np.randomn.randint(1,85)
  pz = []
  simLen = 10*1000
  noise = True
  forwardSim(ez,pz,noise,simLen)

