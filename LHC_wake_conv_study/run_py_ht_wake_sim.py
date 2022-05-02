from py_ht_wake_sim import py_ht_wake_sim

import os

n_sl = int(os.environ['N_SL'])
n_mp = int(os.environ['N_MP'])

print('n_sl=' + str(n_sl))
print('n_mp=' + str(n_mp))

#n_mp_p_sl = 500
py_ht_wake_sim(n_mp, n_sl, 'fig.png')
