# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
step_module = tf.load_op_library('./eulerstep_2d_epileptor_single.so')
# %% Run a simulation
theta = tf.constant(-1.5, dtype=tf.float32, shape=(1,))
y_init = tf.constant([-2.0, 4.8], dtype=tf.float32)
nsteps = 500
y = tf.TensorArray(dtype=tf.float32, size=nsteps, clear_after_read=False)
for i in range(nsteps):
    if(i == 0):
        y_next = y_init
    else:
        y_next = step_module.euler_step2d_epileptor_single(theta, y_next)
        # y_next = euler_step2d_epileptor_single(theta, y_next)
    y = y.write(i, y_next)
y = y.stack()
# %%
plt.figure(figsize=(25,10))
plt.subplot(211)
plt.plot(y[:,0])
plt.ylabel('x')
plt.subplot(212)
plt.plot(y[:,1])
plt.ylabel('z')