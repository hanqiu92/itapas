import numpy as np
import time
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import graph_tool.all as gt
from itapas import ITAPAS
        
# link cost functions
def sys_cost_func(x,param):
    return (param[0] * x + param[1] * (x ** 5),
            param[0] + 5 * param[1] * (x ** 4),
            20 * param[1] * (x ** 3))

def user_cost_func(x,param):
    return (param[0] * x + 0.2 * param[1] * (x ** 5),
            param[0] + param[1] * (x ** 4),
            4 * param[1] * (x ** 3))

# read & process input
# network format: ID, origin node, destination node, param0, param1, background flow
network = np.loadtxt('data/network.csv',delimiter=',',dtype=None)
# initiate network information
link_graph = gt.Graph()
link_graph.add_edge_list(network[:,[1,2]].astype(int))
link_param0 = link_graph.new_edge_property("float")
link_param1 = link_graph.new_edge_property("float")
link_background = link_graph.new_edge_property("float")
link_param0.a = network[:,3]
link_param1.a = network[:,4]
link_background.a = network[:,5]
link_param = [link_param0,link_param1] # param for link cost functions

# od format: ID, home node, work node
ods = np.loadtxt('data/ods.csv',delimiter=',',dtype=int)
# initiate node list
node_list = set()
for od in ods:
    node_list.update([od[1],od[2]])
node_list = list(node_list)

# initiate assignment model
assign_model = ITAPAS(link_graph,link_param,
                    link_background,link_cost_func=user_cost_func,ori_list=node_list)

# initiate demand
scale = 10
demand = dict()
for od in ods:
    if od[1] != od[2]:
        demand[(od[1],od[2])] = demand.get((od[1],od[2]),0) + scale

# do assignment
itapas_params = {'epsilon':1e-12,'theta':1e-12,'mu':1e-3,'nu':0.5,
                 'time':1200,'iter':20,'out_flag':True}
itapas_params['dg'] = np.sum(list(demand.values())) * 0.1
link_flow,link_flow_node = assign_model.assign(demand,params=itapas_params)

