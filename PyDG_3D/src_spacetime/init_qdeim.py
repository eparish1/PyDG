import numpy as np


def init_stencil_qdeim(region,eqns,order):
    data = np.load('rhs_pod_basis.npz')
    region.cell_list = data['index_global']
    region.cell_ijk = np.unravel_index([region.cell_list], (eqns.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt))
    ### make a new variable which only goes through elements (not quadpoint)
    region.cell_ijk2 = np.copy(region.cell_ijk)
    region.cell_ijk2[0][0][:] = 0
    region.cell_ijk2[1][0][:] = 0
    region.cell_ijk2[2][0][:] = 0
    region.cell_ijk2[3][0][:] = 0
    region.cell_ijk2[4][0][:] = 0
    cell_ijk2 = region.cell_ijk2
    region.cell_ijk2 = np.ravel_multi_index( (cell_ijk2[0,0,:],cell_ijk2[1,0,:],cell_ijk2[2,0,:],cell_ijk2[3,0,:],cell_ijk2[4,0,:],cell_ijk2[5,0,:],cell_ijk2[6,0,:],cell_ijk2[7,0,:],cell_ijk2[8,0,:]),(eqns.nvars,order[0],order[1],order[2],order[3],region.Npx,region.Npy,region.Npz,region.Npt))
    ## Now delete any repeated indices
    region.cell_ijk2 = np.unique(region.cell_ijk2)
    region.cell_ijk2 = np.unravel_index([region.cell_ijk2], (eqns.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt)) 
    region.cell_ijk = np.copy(region.cell_ijk2) #set cell_ijk to be the new variable

    stencil_ijk = np.copy(region.cell_ijk)

    cell_tmp = np.copy(region.cell_ijk)
    #cell_tmp[5][0][:] = np.fmin(cell_tmp[5][0][:] + 1 , region.Npx - 1 )
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] + 1)%region.Npx
    stencil_ijk = np.append(stencil_ijk,cell_tmp,axis=2)

    cell_tmp = np.copy(region.cell_ijk)
    #cell_tmp[5][0][:] = np.fmax(cell_tmp[5][0][:] - 1 , 0 )
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] - 1)%region.Npx
    stencil_ijk = np.append(stencil_ijk,cell_tmp,axis=2)

    cell_tmp = np.copy(region.cell_ijk)
    #cell_tmp[6][0][:] = np.fmin(cell_tmp[6][0][:] + 1 , region.Npy - 1)
    cell_tmp[6][0][:] = (cell_tmp[6][0][:] + 1)%region.Npy
    stencil_ijk = np.append(stencil_ijk,cell_tmp,axis=2)

    cell_tmp = np.copy(region.cell_ijk)
    #cell_tmp[6][0][:] = np.fmax(cell_tmp[6][0][:] - 1 , 0   )
    cell_tmp[6][0][:] = (cell_tmp[6][0][:] - 1)%region.Npy
    stencil_ijk = np.append(stencil_ijk,cell_tmp,axis=2)

    cell_tmp = np.copy(region.cell_ijk)
    #cell_tmp[7][0][:] = np.fmin(cell_tmp[7][0][:] + 1 , region.Npz - 1 )
    cell_tmp[7][0][:] = (cell_tmp[7][0][:] + 1)%region.Npz
    stencil_ijk = np.append(stencil_ijk,cell_tmp,axis=2)

    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[7][0][:] = np.fmax(cell_tmp[7][0][:] - 1 , 0   )
    stencil_ijk = np.append(stencil_ijk,cell_tmp,axis=2)


    ## NEED TO DO plus minus two for BR1 diffusion scheme, plus extra stencil!! This sucks!
    cell_tmp = np.copy(region.cell_ijk)
    #cell_tmp[5][0][:] = np.fmin(cell_tmp[5][0][:] + 2 , region.Npx - 1 )
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] + 2)%region.Npx
    viscous_stencil_ijk = np.append(stencil_ijk,cell_tmp,axis=2)

    cell_tmp = np.copy(region.cell_ijk)
    #cell_tmp[5][0][:] = np.fmax(cell_tmp[5][0][:] - 2 , 0 )
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] - 2)%region.Npx
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)

    cell_tmp = np.copy(region.cell_ijk)
    #cell_tmp[6][0][:] = np.fmin(cell_tmp[6][0][:] + 2 , region.Npy - 1)
    cell_tmp[6][0][:] = (cell_tmp[6][0][:] + 2)%region.Npy
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)

    cell_tmp = np.copy(region.cell_ijk)
    #cell_tmp[6][0][:] = np.fmax(cell_tmp[6][0][:] - 2 , 0   )
    cell_tmp[6][0][:] = (cell_tmp[6][0][:] - 2)%region.Npy
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)

    cell_tmp = np.copy(region.cell_ijk)
    #cell_tmp[7][0][:] = np.fmin(cell_tmp[7][0][:] + 2 , region.Npz - 1 )
    cell_tmp[7][0][:] = (cell_tmp[7][0][:] + 2)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)

    cell_tmp = np.copy(region.cell_ijk)
    #cell_tmp[7][0][:] = np.fmax(cell_tmp[7][0][:] - 2 , 0   )
    cell_tmp[7][0][:] = (cell_tmp[7][0][:] - 2)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)

    ## now corners
    ## first do x y corners
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] + 1)%region.Npx
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] + 1)%region.Npy
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] - 1)%region.Npx
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] + 1)%region.Npy
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] + 1)%region.Npx
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] - 1)%region.Npy
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] - 1)%region.Npx
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] - 1)%region.Npy
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)

    ## now do x z corners
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] + 1)%region.Npx
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] + 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] - 1)%region.Npx
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] + 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] + 1)%region.Npx
    cell_tmp[7][0][:] =  (cell_tmp[6][0][:] - 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] - 1)%region.Npx
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] - 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)

    ## now do y z corners
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] + 1)%region.Npy
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] + 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] - 1)%region.Npy
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] + 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] + 1)%region.Npy
    cell_tmp[7][0][:] =  (cell_tmp[6][0][:] - 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] - 1)%region.Npy
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] - 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)

    ## finally do x y z corners
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] + 1)%region.Npx
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] + 1)%region.Npy
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] + 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp = np.copy(region.cell_ijk)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] - 1)%region.Npx
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] + 1)%region.Npy
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] + 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] + 1)%region.Npx
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] - 1)%region.Npy
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] + 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] + 1)%region.Npx
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] + 1)%region.Npy
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] - 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] + 1)%region.Npx
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] - 1)%region.Npy
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] - 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] - 1)%region.Npx
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] + 1)%region.Npy
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] - 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] - 1)%region.Npx
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] - 1)%region.Npy
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] + 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)
    cell_tmp[5][0][:] =  (cell_tmp[5][0][:] - 1)%region.Npx
    cell_tmp[6][0][:] =  (cell_tmp[6][0][:] - 1)%region.Npy
    cell_tmp[7][0][:] =  (cell_tmp[7][0][:] - 1)%region.Npz
    viscous_stencil_ijk = np.append(viscous_stencil_ijk,cell_tmp,axis=2)



    ## now make the reconstruct stencil. This should have all quad points and all conserved variables
    ## ADD ALL CONSERVED VARIABLES TO RECONSTRUCT STENCIL
#    stencil_ijk0 = np.copy(stencil_ijk)
#    viscous_stencil_ijk0 = np.copy(viscous_stencil_ijk)
    if (eqns.vflux_type == 'Inviscid'):
      print('Running inviscid, not doing extended BR1 stencil')
      viscous_stencil_ijk = stencil_ijk

    rec_stencil_ijk0 = np.copy(viscous_stencil_ijk)
    rec_stencil_ijk = np.copy(viscous_stencil_ijk)

    for i in range(0,eqns.nvars):
#       tmp = np.copy(stencil_ijk0)
#       tmp[0][0][:] = i
#       stencil_ijk = np.append(stencil_ijk,tmp,axis=2)
#       tmp = np.copy(viscous_stencil_ijk0)
#       tmp[0][0][:] = i
#       viscous_stencil_ijk = np.append(viscous_stencil_ijk,tmp,axis=2)
       tmp = np.copy(rec_stencil_ijk0)
       tmp[0][0][:] = i
       rec_stencil_ijk = np.append(rec_stencil_ijk,tmp,axis=2)

    rec_stencil_ijk0 = np.copy(rec_stencil_ijk)
    for i in range(0,order[0]):
      for j in range(0,order[1]):
        for k in range(0,order[2]):
          for l in range(0,order[3]): 
            tmp = np.copy(rec_stencil_ijk0)
            tmp[1][0][:] = i
            tmp[2][0][:] = j
            tmp[3][0][:] = k
            tmp[4][0][:] = l
            rec_stencil_ijk = np.append(rec_stencil_ijk,tmp,axis=2)



    region.stencil_ijk = stencil_ijk
    region.stencil_list = np.ravel_multi_index( (stencil_ijk[0,0,:],stencil_ijk[1,0,:],stencil_ijk[2,0,:],stencil_ijk[3,0,:],stencil_ijk[4,0,:],stencil_ijk[5,0,:],stencil_ijk[6,0,:],stencil_ijk[7,0,:],stencil_ijk[8,0,:]),(eqns.nvars,order[0],order[1],order[2],order[3],region.Npx,region.Npy,region.Npz,region.Npt))
    ## Now delete any repeated indices
    region.stencil_list = np.unique(region.stencil_list)
    # Now recreate call to indices
    region.stencil_ijk = np.unravel_index([region.stencil_list], (eqns.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt))

    region.rec_stencil_ijk = rec_stencil_ijk
    region.rec_stencil_list = np.ravel_multi_index( (rec_stencil_ijk[0,0,:],rec_stencil_ijk[1,0,:],rec_stencil_ijk[2,0,:],rec_stencil_ijk[3,0,:],rec_stencil_ijk[4,0,:],rec_stencil_ijk[5,0,:],rec_stencil_ijk[6,0,:],rec_stencil_ijk[7,0,:],rec_stencil_ijk[8,0,:]),(eqns.nvars,order[0],order[1],order[2],order[3],region.Npx,region.Npy,region.Npz,region.Npt))
    ## Now delete any repeated indices
    region.rec_stencil_list = np.unique(region.rec_stencil_list)
    #region.rec_stencil_list = range(0,eqns.nvars*region.order[0]*region.order[1]*region.order[2]*region.order[3]*region.Npx*region.Npy*region.Npz*region.Npt)
    # Now recreate call to indices
    region.rec_stencil_ijk = np.unravel_index([region.rec_stencil_list], (eqns.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt))


    region.viscous_stencil_ijk = viscous_stencil_ijk
    region.viscous_stencil_list = np.ravel_multi_index( (viscous_stencil_ijk[0,0,:],viscous_stencil_ijk[1,0,:],viscous_stencil_ijk[2,0,:],viscous_stencil_ijk[3,0,:],viscous_stencil_ijk[4,0,:],viscous_stencil_ijk[5,0,:],viscous_stencil_ijk[6,0,:],viscous_stencil_ijk[7,0,:],viscous_stencil_ijk[8,0,:]),(eqns.nvars,order[0],order[1],order[2],order[3],region.Npx,region.Npy,region.Npz,region.Npt))
    ## Now delete any repeated indices
    region.viscous_stencil_list = np.unique(region.viscous_stencil_list)
    # Now recreate call to indices
    region.viscous_stencil_ijk = np.unravel_index([region.viscous_stencil_list], (eqns.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt))
    #region.RM = data['M']
    cell_ijk = region.cell_ijk
    stencil_ijk = region.stencil_ijk

    region.iFlux.fx_hyper = np.zeros(np.shape(region.iFlux.fx[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]),dtype=region.iFlux.fx.dtype)
    region.iFlux.fy_hyper = np.zeros(np.shape(region.iFlux.fy[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]),dtype=region.iFlux.fy.dtype)
    region.iFlux.fz_hyper = np.zeros(np.shape(region.iFlux.fz[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]),dtype=region.iFlux.fz.dtype)

    region.vFlux.fx_hyper = np.zeros(np.shape(region.vFlux.fx[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]),dtype=region.vFlux.fx.dtype)
    region.vFlux.fy_hyper = np.zeros(np.shape(region.vFlux.fy[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]),dtype=region.vFlux.fy.dtype)
    region.vFlux.fz_hyper = np.zeros(np.shape(region.vFlux.fz[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]),dtype=region.vFlux.fz.dtype)
    region.vFlux2.fx_hyper = np.zeros(np.shape(region.vFlux2.fx[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]),dtype=region.vFlux2.fx.dtype)
    region.vFlux2.fy_hyper = np.zeros(np.shape(region.vFlux2.fy[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]),dtype=region.vFlux2.fy.dtype)
    region.vFlux2.fz_hyper = np.zeros(np.shape(region.vFlux2.fz[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]),dtype=region.vFlux2.fz.dtype)

    region.iFlux.fRI_hyper = np.zeros(np.shape(region.iFlux.fRI[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) , dtype=region.iFlux.fRI.dtype) 
    region.iFlux.fLI_hyper = np.zeros(np.shape(region.iFlux.fLI[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) , dtype=region.iFlux.fLI.dtype) 
    region.iFlux.fUI_hyper = np.zeros(np.shape(region.iFlux.fUI[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) , dtype=region.iFlux.fUI.dtype)
    region.iFlux.fDI_hyper = np.zeros(np.shape(region.iFlux.fDI[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) , dtype=region.iFlux.fDI.dtype)
    region.iFlux.fFI_hyper = np.zeros(np.shape(region.iFlux.fFI[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) , dtype=region.iFlux.fFI.dtype) 
    region.iFlux.fBI_hyper = np.zeros(np.shape(region.iFlux.fBI[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) , dtype=region.iFlux.fBI.dtype) 

    region.vFlux.fRI_hyper = np.zeros(np.shape(region.vFlux.fRI[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] ) , dtype=region.vFlux.fRI.dtype) 
    region.vFlux.fLI_hyper = np.zeros(np.shape(region.vFlux.fLI[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] ) , dtype=region.vFlux.fLI.dtype) 
    region.vFlux.fUI_hyper = np.zeros(np.shape(region.vFlux.fUI[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] ) , dtype=region.vFlux.fUI.dtype)
    region.vFlux.fDI_hyper = np.zeros(np.shape(region.vFlux.fDI[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] ) , dtype=region.vFlux.fDI.dtype)
    region.vFlux.fFI_hyper = np.zeros(np.shape(region.vFlux.fFI[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] ) , dtype=region.vFlux.fFI.dtype) 
    region.vFlux.fBI_hyper = np.zeros(np.shape(region.vFlux.fBI[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] ) , dtype=region.vFlux.fBI.dtype) 


    region.RHS_hyper = np.zeros(np.shape(region.RHS[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]),dtype=region.RHS.dtype)



    region.J_edge_det_hyper_x1p1 = region.J_edge_det[0][:,:,cell_ijk[5][0]+1,cell_ijk[6][0],cell_ijk[7][0]]
    region.J_edge_det_hyper_x1 = region.J_edge_det[0][:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
    region.J_edge_det_hyper_x2p1 = region.J_edge_det[1][:,:,cell_ijk[5][0],cell_ijk[6][0]+1,cell_ijk[7][0]]
    region.J_edge_det_hyper_x2 = region.J_edge_det[1][:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]
    region.J_edge_det_hyper_x3p1 = region.J_edge_det[2][:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]+1]
    region.J_edge_det_hyper_x3 = region.J_edge_det[2][:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]]


    region.iFlux.fRS_hyper = np.zeros(np.shape( region.iFlux.fRS[:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=region.iFlux.fRS.dtype)
    #region.iFlux.fRS_hyper_edge = np.zeros(np.shape( region.iFlux.fRS[:,:,:,:,-1,cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] ) ,dtype=region.iFlux.fRS.dtype)










