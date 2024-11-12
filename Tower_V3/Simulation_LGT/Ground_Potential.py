def Ground_Potential(CK_Para, Node, GND):

    ep0 = 8.854187818e-12
    epr = GND['epr']
    k = ep0 / GND['sig']

    P = CK_Para['P']
    G = CK_Para['G']

    ns = Node['num'][3] + 1  # starting node id
    ne = Node['num'][3] + Node['num'][2]  # ending node id
    nn = Node['num'][2]  # ground node #

    Pg = P[ns:ne, ns:ne]  # potential coefficients of ground node

    G[ns:ne, ns:ne] = k * Pg  # ground conductance
    P[ns:ne, ns:ne] = Pg / epr  # ground potential

    return G, P