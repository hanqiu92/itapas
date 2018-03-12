import numpy as np
import time
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import graph_tool.all as gt

class ITAPAS(object):

    def __init__(self,network,link_param,link_background,link_cost_func,ori_list=None):
        '''
        # input
        # network: gt Graph object
        # link_param: list of gt PropertyMap object on links, as input for link_cost_func
        # link_background: gt PropertyMap object on links, for background traffic volume
        # link_cost_func: cost function generalizing link volume-travel-time relationship
        # ori_list: list of vertices in the Graph network
        '''
        self.network = network
        self.link_param = link_param
        self.link_background = link_background
        self.link_cost_func = link_cost_func

        # get the edge list sorted for fast manipulation of edge properties
        edge_list = np.array(network.get_edges(),dtype=np.int32)
        ind = np.argsort(edge_list[:,2])
        edge_list = edge_list[ind,:]
        edge_index = {}
        for i in xrange(edge_list.shape[0]):
            edge_index[(edge_list[i,0],edge_list[i,1])] = i

        self.edge_list = edge_list
        self.edge_index = edge_index

        if not ori_list:
            # use the list of all nodes
            ori_list = list(self.network.get_vertices())
        ori_index = {}
        for idx,key in enumerate(ori_list):
            ori_index[key] = idx
        self.ori_index = ori_index

    def MFS(self,flow_map,local_flow_map,cost_map,cost_de_map,
            prev_map,r,e0,epsilon):
        network = self.network
        param_map = self.link_param
        link_background = self.link_background
        link_cost_func = self.link_cost_func
        edge_index = self.edge_index

        prev_new_map = network.new_vertex_property("int",val=-1)
        status_map = network.new_vertex_property("int",val=0)
        v_list,e_list = gt.shortest_path(network,source=network.vertex(r),
                                        target=network.vertex(e0[1]),pred_map=prev_map)
        if e0[0] == r:
            return (tuple([edge_index.get(tuple(e)) for e in e_list]),tuple([edge_index.get(e0)]))

        # count = 0
        # debug_flag = False
        while True:
            # if count > 100:
            #     debug_flag = True
            #     print count,r,e0,local_flow_map[e0],[int(v) for v in v_list]
            # step 1: initialization
            prev_new_map.set_value(-1)
            status_map.set_value(0)
            i,j = e0
            prev_new_map[j] = i
            for v in v_list:
                status_map[v] = -1
            status_map[j] = 1
            status_map[i] = 1

            inner_flag = True
            while inner_flag:
                # step 2: find max incoming edge
                m_list = [int(m) for m in network.vertex(i).in_neighbours()]
                f_list = [local_flow_map[(m,i)] for m in m_list]
                m = m_list[np.argmax(f_list)]
                # if debug_flag:
                #     print i,m_list,f_list,status_map[m]

                prev_new_map[i] = m
                if status_map[m] == -1:
                    # step 3: return pas
                    _,e1 = gt.shortest_path(network,source=network.vertex(m),
                                        target=network.vertex(j),pred_map=prev_map)
                    _,e2 = gt.shortest_path(network,source=network.vertex(m),
                                        target=network.vertex(j),pred_map=prev_new_map)
                    return (tuple([edge_index.get(tuple(e)) for e in e1]),
                            tuple([edge_index.get(tuple(e)) for e in e2]))
                elif status_map[m] == 1:
                    # step 4: remove cycle
                    e_list = []
                    curr_m = prev_new_map[m]
                    e_list += [edge_index.get((curr_m,m))]
                    prev_m = prev_new_map[curr_m]
                    while curr_m != m:
                        e_list += [edge_index.get((prev_m,curr_m))]
                        curr_m = prev_m
                        prev_m = prev_new_map[curr_m]

                    delta = np.min(local_flow_map.a[e_list])
                    local_flow_map.a[e_list] -= delta
                    flow_map.a[e_list] -= delta
                    _,cost_map.a[e_list],cost_de_map.a[e_list] = link_cost_func(flow_map.a[e_list] + \
                                                                                  link_background.a[e_list],
                                                                              [p.a[e_list] for p in param_map])
                    if local_flow_map[e0] < epsilon:
                        return None
                    inner_flag = False
                    # if debug_flag:
                    #     print delta
                else:
                    i = m
                    status_map[i] = 1
                # count += 1

    def shift(self,link_flow,local_link_flow,link_cost,link_cost_de,pas,epsilon,mu,init_flag):
        link_param = self.link_param
        link_background = self.link_background
        cost_func = self.link_cost_func

        ind_e1 = list(pas[0])
        ind_e2 = list(pas[1])
        f1 = float(np.min(local_link_flow.a[ind_e1]))
        f2 = float(np.min(local_link_flow.a[ind_e2]))
        t1 = np.sum(link_cost.a[ind_e1])
        t2 = np.sum(link_cost.a[ind_e2])
        if f1 < epsilon and f2 < epsilon:
            return (0,0)
        elif (f1 < epsilon or f2 < epsilon or np.abs(t2-t1) < mu*(t2+t1)) and not init_flag:
            return None
        else:
            dt1 = np.sum(link_cost_de.a[ind_e1])
            dt2 = np.sum(link_cost_de.a[ind_e2])
            delta = float((t2-t1) / (dt1+dt2))
            delta = min(delta,f2) if (delta > 0) else (max(delta,-f1))

            local_link_flow.a[ind_e1] += delta
            link_flow.a[ind_e1] += delta
            local_link_flow.a[ind_e2] -= delta
            link_flow.a[ind_e2] -= delta
            ind_e = ind_e1 + ind_e2
            _,link_cost.a[ind_e],link_cost_de.a[ind_e] = cost_func(link_flow.a[ind_e] + link_background.a[ind_e],
                                                                 [p.a[ind_e] for p in link_param])
            return (f1+delta,f2-delta)

    def get_duality_gap(self,link_flow,link_cost,demand):
        network = self.network
        primal = float(np.sum(link_flow.a * link_cost.a))
        dual = 0.0
        for ori in self.ori_index.keys():
            d_map = gt.shortest_distance(network, source=network.vertex(ori), weights=link_cost, max_dist=100000)
            for key,value in demand.iteritems():
                if key[0] == ori:
                    dual += value * d_map[key[1]]
        dg = primal - dual
        return dg

    def assign(self,demand,init_flow = None,init_flow_node = None,
               params = {'epsilon':1e-12,'theta':1e-12,'mu':1e-3,'nu':0.5,
                         'dg':10,'time':1200,'iter':50,'out_flag':False}):
        '''
        # input
        # demand: python dictionary of demand record in form {(o_node,d_node):flow}
        # init_flow: initial flow assignment of the demand
        # params: parameter for the algorithm
        # out_flag: bool variable for output
        '''
        t = time.time()
        epsilon = params['epsilon']
        theta = params['theta']
        mu = params['mu']
        nu = params['nu']
        dg_limit = params['dg']
        max_iter = params['iter']
        time_limit = params['time']
        out_flag = params['out_flag']

        network = self.network
        link_background = self.link_background
        link_param = self.link_param
        cost_func = self.link_cost_func
        edge_list = self.edge_list
        edge_index = self.edge_index
        ori_index = self.ori_index

        link_cost = network.new_edge_property("float",val=0)
        link_cost_de = network.new_edge_property("float",val=0)
        link_flow = network.new_edge_property("float",val=0)
        link_flow_node = {}
        for ori in ori_index.keys():
            link_flow_node[ori] = network.new_edge_property("float",val=0)

        # Step 0 : Initialization
        if not init_flow or not init_flow_node:
            _,link_cost.a,_ = cost_func(link_background.a,[p.a for p in link_param])

            subnet = {}
            for ori in ori_index.keys():
                _,subnet[ori] = gt.shortest_distance(network, source=network.vertex(ori), weights=link_cost,
                                                     max_dist=100000, pred_map=True)
            for key,value in demand.iteritems():
                _,e_list = gt.shortest_path(network, source=network.vertex(key[0]), target=network.vertex(key[1]),
                                            pred_map=subnet[key[0]])
                e_list = [edge_index.get(tuple(e)) for e in e_list]
                link_flow.a[e_list] += value
                link_flow_node[key[0]].a[e_list] += value
        else:
            link_flow.a = init_flow.a
            for ori in ori_index.keys():
                link_flow_node[ori].a = init_flow_node[ori].a

        _,link_cost.a,link_cost_de.a = cost_func(link_flow.a + link_background.a,
                                               [p.a for p in self.link_param])
        dg = self.get_duality_gap(link_flow,link_cost,demand)
        if out_flag:
            print 'ITAPAS: Initial duality gap = {:.0f}.  Elapsed time = {:.2f}.'.format(dg,time.time() - t)
        if dg < dg_limit:
            return link_flow,link_flow_node

        pas_set = {} # (e1,e2):r0
        pas_set_node = {}
        for v in network.get_vertices():
            pas_set_node[int(v)] = {}

        flag = True
        curr_iter = 0
        while flag:
            comp_slack = 0.0
            for ori in ori_index.keys():
                # Step 1:
                local_link_flow = link_flow_node[ori]
                d_map,prev_map = gt.shortest_distance(network, source=network.vertex(ori), weights=link_cost,
                                                     max_dist=10000, pred_map=True)

                # update reduced cost and put into link set node
                link_set = []
                rc_start = d_map.a[edge_list[:,0]]
                rc_start[np.isinf(rc_start)] = -1
                rc_end = d_map.a[edge_list[:,1]]
                rc_end[np.isinf(rc_end)] = 100000
                comp_slack = float(np.sum(local_link_flow.a * (rc_start - rc_end + link_cost.a)))
                if comp_slack > 1e-8:
                    ind_list = list(np.logical_and(local_link_flow.a > epsilon * comp_slack,
                                                   (rc_start - rc_end + link_cost.a) > theta * comp_slack))
                    link_set = [(edge_list[i,0],edge_list[i,1]) for i,ind in enumerate(ind_list) if ind]

                while link_set:
                    # identify pas
                    e = link_set.pop()
                    if local_link_flow[e] > epsilon:
                        pas = self.MFS(link_flow,local_link_flow,link_cost,link_cost_de,
                                       prev_map,ori,e,epsilon)
                        if pas:
                            if pas not in pas_set:
                                pas_set[pas] = ori
                                pas_set_node[e[1]][pas] = ori

                if pas_set:
                    pas_list = pas_set.keys()
                    if len(pas_list) > 100:
                        pas_list_select = [pas_list[idx] for idx in np.random.choice(len(pas_list),100,replace=False)]
                    else:
                        pas_list_select = pas_list
                    for pas in pas_list_select:
                        self.shift(link_flow,link_flow_node[pas_set.get(pas)],
                                  link_cost,link_cost_de,pas,epsilon,mu,True)

            for _ in xrange(20):
                pas_list = pas_set.keys()
                for pas in pas_list:
                    if not self.shift(link_flow,link_flow_node[pas_set.get(pas)],
                                      link_cost,link_cost_de,pas,epsilon,mu,False):
                        del pas_set[pas]

            _,link_cost.a,link_cost_de.a = cost_func(link_flow.a + link_background.a,
                                               [p.a for p in self.link_param])
            dg = self.get_duality_gap(link_flow,link_cost,demand)
            if out_flag:
                print 'ITAPAS: Iteration {} finished. PAS size = {}. Duality gap = {:.0f}. Elapsed time = {:.2f}.'.format(curr_iter,len(pas_set),dg,time.time() - t)
            curr_iter += 1
            if (dg < dg_limit) or (curr_iter > max_iter) or (time.time() - t > time_limit):
                return link_flow,link_flow_node
