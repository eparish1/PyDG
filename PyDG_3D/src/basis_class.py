import numpy as np
from mpi4py import MPI
from tensor_products import *
class basis_class:
  def __init__(self,basis_str,args):
    comm = MPI.COMM_WORLD
    self.basis_str = basis_str
    mpi_rank = comm.Get_rank()  
    if (mpi_rank == 0): print('Using ' + basis_str + ' basis' )
    if (args[0] == 'TensorDot'):
      if (mpi_rank == 0): print('Using ' + args[0] + ' function for modal computations')
      self.diffU = diffU_tensordot
      self.diffUXEdge_edge = diffUXEdge_edge_tensordot
      self.diffUX_edge = diffUX_edge_tensordot
      self.diffUYEdge_edge = diffUYEdge_edge_tensordot
      self.diffUY_edge = diffUY_edge_tensordot
      self.diffUZEdge_edge = diffUZEdge_edge_tensordot
      self.diffUZ_edge = diffUZ_edge_tensordot
      self.volIntegrateGlob = volIntegrateGlob_tensordot
      self.faceIntegrateGlob = faceIntegrateGlob_tensordot
      self.reconstructU = reconstructU_tensordot
      self.reconstructEdgesGeneral = reconstructEdgesGeneral_tensordot
      self.reconstructUGeneral = reconstructUGeneral_tensordot
      self.reconstructEdgeEdgesGeneral = reconstructEdgeEdgesGeneral_tensordot

    if (args[0] == 'einsum'):
      if (mpi_rank == 0): print('Using ' + args[0] + ' function for modal computations')
      self.diffU = diffU_einsum
      self.diffUXEdge_edge = diffUXEdge_edge_einsum
      self.diffUX_edge = diffUX_edge_einsum
      self.diffUYEdge_edge = diffUYEdge_edge_einsum
      self.diffUY_edge = diffUY_edge_einsum
      self.diffUZEdge_edge = diffUZEdge_edge_einsum
      self.diffUZ_edge = diffUZ_edge_einsum
      self.volIntegrateGlob = volIntegrateGlob_einsum
      self.faceIntegrateGlob = faceIntegrateGlob_einsum
      self.reconstructU = reconstructU_einsum
      self.reconstructEdgesGeneral = reconstructEdgesGeneral_einsum
      self.reconstructUGeneral = reconstructUGeneral_einsum
      self.reconstructEdgeEdgesGeneral = reconstructEdgeEdgesGeneral_einsum
