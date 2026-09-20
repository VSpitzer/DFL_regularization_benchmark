import time
import numpy as np
import json
import pickle
import copy
import sys


# Problem classes whose scale parameter b doesn't follow the b = 10**(instance-1)
# pattern used for instances 1/3 (b = 1/100). Checked first in __init__; any
# instance not listed here falls back to that formula.
#
# The three problem classes actually studied are instances 1/2/3, for
# b = 1/25/100 respectively -- instance 2 is the odd one out (it used to be
# b = 10**(2-1) = 10, but that instance was dropped; b = 25 was slotted into
# the "2" label in its place, and is the only override needed).
INSTANCE_B_OVERRIDES = {
    2: 25,
}


class InstabilityProblem:
    def __init__(self, instance) -> None:
        if instance in INSTANCE_B_OVERRIDES:
            self.b = INSTANCE_B_OVERRIDES[instance]
        else:
            self.b = 10**(instance-1)

    def solve(self, preds, data):
        theta, = data
        tab_vertex = [[np.cos(0),np.sin(0)],[self.b*np.cos(2*np.pi/3),self.b*np.sin(2*np.pi/3)],[np.cos(4*np.pi/3),np.sin(4*np.pi/3)]]
        sol_set = np.array([ rotate(theta,v) for v in tab_vertex])
        id_sol = np.argmax( np.dot(sol_set,preds))
        return sol_set[id_sol]

    def get_vertex_solve(self, preds, data):
        theta, = data
        tab_vertex = [[np.cos(0),np.sin(0)],[self.b*np.cos(2*np.pi/3),self.b*np.sin(2*np.pi/3)],[np.cos(4*np.pi/3),np.sin(4*np.pi/3)]]
        sol_set = np.array([ rotate(theta,v) for v in tab_vertex])
        id_sol = np.argmax( np.dot(sol_set,preds))
        return id_sol

def rotate(theta,vec):
    return np.dot(np.array([[np.cos(theta),-np.sin(theta)],[np.cos(theta),np.sin(theta)]]),np.array(vec))

def generate_data(N):
    data = dict()
    tab_eps = (np.random.rand(N)*np.tan(np.pi*1./3)).tolist()
    tab_rot = (np.random.rand(N)*2*np.pi).tolist()

    data['inputs'] = [ [tab_rot[i]] for i in range(N)]
    data['outputs'] = [rotate(tab_rot[i],[tab_eps[i],-1]).tolist() for i in range(N)]
    data['params'] = [ [tab_rot[i]] for i in range(N)]
    with open('data/data_instability_problem.json', 'w') as f:
        json.dump(data, f)

# generate_data(300)
