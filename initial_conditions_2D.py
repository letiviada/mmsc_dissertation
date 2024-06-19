import numpy

def four_reg_prescribed(num_nodes: int, num_refs: int):
    """
    Make a four-regular cell with manually prescribed edge conductances 
    for num_nodes = 1,4,9, for testing.

    Parameters 
    ----------

    - num_nodes: int
        Number of nodes in the cell.
    - num_refs: int
        Number of lengths in the reference set. 
        For example, if reference set is r=[-1,0,+1] then num_refs==3.
    
    Returns 
    -------
    - cond_init_4: numpy.ndarray 
        cond_init_4[i,j,r,s] is the initial conductance of the edge 
        between node i in one cell and node j in the cell at position r,s relative 
        to the first cell.
    """
    
    # Define parameters 
    # -----
    cond_init_4 = numpy.zeros(shape=(num_nodes, num_nodes, num_refs, num_refs))
    
    if num_nodes == 1:

        # Grid of one node
        # ----------------
        cond_init_4[0,0,1,0]  = 1
        cond_init_4[0,0,-1,0] = 1

        cond_init_4[0,0,0,1]  = 1
        cond_init_4[0,0,0,-1] = 1


    elif num_nodes == 4:
        
        # Grid of four nodes
        #--------------------
        #           2         3 
        #           |         |
        #         (1.0)     (1.0)
        #           |         |
        # 1--(1.0)--0--(1.0)--1--(1.0)--0
        #           |         |
        #         (1.0)     (1.0)
        #           |         |
        # 3--(1.0)--2--(1.0)--3--(1.0)--2
        #           |         |
        #         (1.0)     (1.0)
        #           |         |
        #           0         1

        # Internal edges
        cond_init_4[0,1,0,0] = 1.0
        cond_init_4[1,0,0,0] = 1.0

        cond_init_4[1,3,0,0] = 1.0
        cond_init_4[3,1,0,0] = 1.0

        cond_init_4[2,3,0,0] = 1.0
        cond_init_4[3,2,0,0] = 1.0

        cond_init_4[0,2,0,0] = 1.0
        cond_init_4[2,0,0,0] = 1.0

        ## External edges
        cond_init_4[1,0,1,0]  = 1.0
        cond_init_4[0,1,-1,0] = 1.0

        cond_init_4[3,2,1,0]  = 1.0
        cond_init_4[2,3,-1,0] = 1.0
        
        cond_init_4[0,2,0,1]  = 1.0
        cond_init_4[2,0,0,-1] = 1.0
        
        cond_init_4[1,3,0,1]  = 1.0
        cond_init_4[3,1,0,-1] = 1.0


    elif num_nodes == 9:
        # Internal edges
        cond_init_4[0,1,0,0] = 1.0#0.8 #1.0
        cond_init_4[1,0,0,0] = 1.0#0.8 #1.0

        cond_init_4[1,2,0,0] = 1.0#0.8 #1.0
        cond_init_4[2,1,0,0] = 1.0#0.8 #1.0

        cond_init_4[3,4,0,0] = 1.0#0.8 #1.0
        cond_init_4[4,3,0,0] = 1.0#0.8 #1.0

        cond_init_4[4,5,0,0] = 1.0#0.8 #1.0
        cond_init_4[5,4,0,0] = 1.0#0.8 #1.0
        
        cond_init_4[6,7,0,0] = 1.0#0.8 #1.0
        cond_init_4[7,6,0,0] = 1.0#0.8 #1.0

        cond_init_4[7,8,0,0] = 1.0#0.8 #1.0
        cond_init_4[8,7,0,0] = 1.0#0.8 #1.0

        cond_init_4[0,3,0,0] = 1.0#0.8 #1.0
        cond_init_4[3,0,0,0] = 1.0#0.8 #1.0
        
        cond_init_4[1,4,0,0] = 1.0#0.8 #1.0
        cond_init_4[4,1,0,0] = 1.0#0.8 #1.0
        
        cond_init_4[2,5,0,0] = 1.0#0.8 #1.0
        cond_init_4[5,2,0,0] = 1.0#0.8 #1.0

        cond_init_4[3,6,0,0] = 1.0#0.8 #1.0
        cond_init_4[6,3,0,0] = 1.0#0.8 #1.0

        cond_init_4[4,7,0,0] = 1.0#0.8 #1.0
        cond_init_4[7,4,0,0] = 1.0#0.8 #1.0

        cond_init_4[5,8,0,0] = 1.0#0.8 #1.0
        cond_init_4[8,5,0,0] = 1.0#0.8 #1.0

        ## External edges
        cond_init_4[2,0,1,0]  = 1.0#1.0 #1.0
        cond_init_4[0,2,-1,0] = 1.0#1.0 #1.0

        cond_init_4[5,3,1,0]  = 1.0#1.0 #1.0
        cond_init_4[3,5,-1,0] = 1.0#1.0 #1.0

        cond_init_4[8,6,1,0]  = 1.0#1.0 #1.0
        cond_init_4[6,8,-1,0] = 1.0#1.0 #1.0

        cond_init_4[0,6,0,1]  = 1.0#1.0 #1.0
        cond_init_4[6,0,0,-1] = 1.0#1.0 #1.0

        cond_init_4[1,7,0,1]  = 1.0#1.0 #1.0
        cond_init_4[7,1,0,-1] = 1.0#1.0 #1.0

        cond_init_4[2,8,0,1]  = 1.0#1.0 #1.0
        cond_init_4[8,2,0,-1] = 1.0#1.0 #1.0

    return cond_init_4