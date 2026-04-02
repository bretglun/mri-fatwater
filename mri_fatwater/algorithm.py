import thinqpbo as tq
import numpy as np
from skimage.filters import threshold_otsu
from dataclasses import replace
from itertools import product
from .constants import GYRO


def get_neighbour_indices(neighbourhood, shape, cyclic):
    assert neighbourhood.shape[1] == len(shape)
    assert len(shape) == len(cyclic)
    num_ngb = neighbourhood.shape[0]
    num_voxels = np.prod(shape)
    indices = np.arange(num_voxels).reshape(shape)
    ngb_indices = np.empty(shape=(num_ngb, *shape), dtype=int)
    edge_ngb = np.full(shape=(num_ngb, *shape), fill_value=False)
    for ngb in range(num_ngb):
        for axis, shift in enumerate(neighbourhood[ngb]):
            if cyclic[axis] or shift==0:
                continue
            slices = [slice(None)] * indices.ndim
            if shift < 0:
                slices[axis] = slice(0, -shift)
            if shift > 0:
                slices[axis] = slice(-shift, shape[axis])
            edge_ngb[ngb, *tuple(slices)] = True
        ngb_indices[ngb] = np.roll(indices, shift=-neighbourhood[ngb], axis=range(indices.ndim))
    return ngb_indices.reshape(num_ngb, num_voxels), edge_ngb.reshape(num_ngb, num_voxels)


def QPBO(D, V, ngb_index):
    num_neighbours, num_voxels = ngb_index.shape
    assert V.shape == (4, num_neighbours, num_voxels)
    graph = tq.QPBOFloat()
    graph.add_node(num_voxels)

    for q in range(num_voxels):
        graph.add_unary_term(q, D[0, q], D[1, q])
        for k in range(num_neighbours): # loop over neighbourhood
            p = ngb_index[k, q]
            if any(V[:, k, q]):
                graph.add_pairwise_term(p, q, *V[:, k, q])
    graph.solve()

    label = np.zeros(num_voxels)
    for q in range(num_voxels):
        label[q] = graph.get_label(q)
    return label


def get_updates(max_update):
    updates = [0] * (2 * max_update + 1)  # Update order
    updates[2:len(updates):2] = list(range(1, max_update + 1)) # Even are positive
    updates[1:len(updates):2] = list(range(-1, -max_update - 1, -1)) # Odd are negative
    return updates


def ICM(prev, L, max_update, num_iters, J, V, w, ngb_indices):
    updates = get_updates(max_update)
    current = np.array(prev)
    for iter in range(num_iters):  # ICM iterate
        print(f'{str(iter+1)}, ', end='')
        prev[:] = current[:]
        min_cost = np.full(current.shape, np.inf)

        for update in updates:
            cost = J[(prev + update) % L, range(J.shape[1])] # Unary cost
            # Binary costs:
            for k in range(ngb_indices.shape[0]):
                cost[ngb_indices[k]] += w[k] * V[abs((prev[ngb_indices[k]]  + update) % L - prev)]
                cost += w[k] * V[abs((prev + update) % L - prev[ngb_indices[k]])]
            current[cost < min_cost] = (prev[cost < min_cost] + update) % L
            min_cost[cost < min_cost] = cost[cost < min_cost]
    return current


def find_minima(J, num_minima=1):
    if J.ndim > 2:
        return find_minima(J.reshape(J.shape[0], -1), num_minima=num_minima).reshape((num_minima, *J.shape[1:]))
    minima, voxels = np.where((J < np.roll(J, 1, axis=0)) & (J < np.roll(J, -1, axis=0)))
    sorting = np.lexsort((J[minima, voxels], voxels))
    minima_sorted = minima[sorting]
    voxels_sorted = voxels[sorting]
    unique_voxels, unique_indices, counts = np.unique(voxels_sorted, return_index=True, return_counts=True)
    minima_array = np.zeros((num_minima, J.shape[1]), dtype=int)
    # Fill resulting array with smallest minimum and then record larger minima if available:
    minima_array[:, unique_voxels] = np.tile(minima_sorted[unique_indices], (num_minima, 1))
    for k in range(1, num_minima):
        minima_array[k, unique_voxels[counts > k]] = minima_sorted[unique_indices[counts > k] + k]
    return minima_array


def isotropy(voxelsize):
    volume = np.prod(voxelsize) # hypervolume
    surface = 2 * volume * np.sum(1 / np.array(voxelsize)) # hypersurface
    norm = volume ** (1/len(voxelsize)) # normalize by hypercube sidelength
    return volume / surface / norm


def get_downsampling(shape, voxelsize):
    best_isotropy = 0 # find downsamling which maximizes isotropy
    dims = np.where(np.array(shape)>1)[0] # dims permitting downsampling
    for factors in product([1, 2], repeat=len(dims)):
        if np.prod(factors)==1:
            continue
        new_voxelsize = np.array(voxelsize)[dims] * factors
        iso = isotropy(new_voxelsize)
        if iso > best_isotropy:
            best_isotropy = iso
            best_factors = factors
    factors = tuple(best_factors[dim] if dim in dims else 1 for dim in range(len(shape)))
    coarse_voxelsize = tuple(sz * f for sz, f in zip(voxelsize, factors))
    coarse_shape = tuple(int(np.ceil(n/f)) for n, f in zip(shape, factors))
    return coarse_shape, coarse_voxelsize


def downsample_residual(J, coarse_shape, fine_shape):
    nB0 = J.shape[0]
    factors = tuple(int(np.ceil(fine/coarse)) for fine, coarse in zip(fine_shape, coarse_shape))
    padding = tuple(fine % upsampling_factor for fine, upsampling_factor in zip(fine_shape, factors))
    J_coarse = np.pad(J.reshape(nB0, *fine_shape), 
                      tuple((0, p) for p in (0,) + padding))
    new_shape = (nB0,) + tuple(s for pair in zip(coarse_shape, factors) for s in pair)
    mean_axes = tuple(2 * (np.arange(len(coarse_shape)) + 1))
    J_coarse = J_coarse.reshape(new_shape).mean(axis=mean_axes)
    return J_coarse.reshape(nB0, -1)


def upsample_B0(dB0, coarse_shape, fine_shape):
    upsampled = dB0.reshape(coarse_shape)
    for dim in range(len(coarse_shape)):
        factor = int(np.ceil(fine_shape[dim]/coarse_shape[dim]))
        upsampled = np.repeat(upsampled, factor, axis=dim)
    return upsampled[tuple(slice(0, s) for s in fine_shape)].flatten()


def get_neighbourhood(radius, voxelsize):
    bound = np.floor(radius / np.array(voxelsize)).astype(int)
    candidates = product(*(range(-b, b + 1) for b in bound))
    neighbours = [ngb for ngb in candidates
                  if ngb > (0,) * len(voxelsize) # filter out duplicates in opposite directions and origin
                  and np.linalg.norm(np.array(ngb) * voxelsize) <= radius]
    # make sure immediate neighbours are always included:
    neighbours.extend([immediate for immediate in np.eye(len(voxelsize), dtype=int) if tuple(immediate) not in neighbours])
    return np.array(neighbours)


def calculate_fieldmap(J, V, aPar, shape, voxelsize, cyclic, offresPenalty=0, offresCenter=0):
    A, B = find_minima(J, num_minima=2)
    dB0 = np.array(A)

    # Multiscale recursion
    if dB0.size == 1:  # Trivial case at coarsest level with only one voxel
        print(f'Level {shape}: Trivial case')
        return dB0

    if aPar.multiScale:
        coarse_shape, coarse_voxelsize = get_downsampling(shape, voxelsize)
        J_coarse = downsample_residual(J, coarse_shape, shape)
        # Recursion:
        dB0_coarse = calculate_fieldmap(J_coarse, V, replace(aPar, graphcutLevel=aPar.graphcutLevel-1), coarse_shape, coarse_voxelsize, cyclic, offresPenalty, offresCenter)
        dB0 = upsample_B0(dB0_coarse, coarse_shape, shape)
        print(f'Level {shape}: ')

    # Prepare MRF
    print('Preparing MRF...', end='')
    # Prepare discontinuity costs
    neighbourhood = get_neighbourhood(aPar.neighbourhoodRadius, voxelsize)
    ngb_indices, edge_ngb = get_neighbour_indices(neighbourhood, shape, cyclic)
    num_ngb = ngb_indices.shape[0]
    
    # 2nd derivative of residual function
    # NOTE: No division by square(steplength) since
    # square(steplength) not included in V
    vxls = range(np.prod(shape))
    ddJ = (J[(A+1) % aPar.nB0, vxls] + J[(A-1) % aPar.nB0, vxls] - 2 * J[A, vxls])

    w = np.zeros((ngb_indices.shape))
    for k, ngb in enumerate(neighbourhood):
        distance = np.linalg.norm(ngb * voxelsize)
        ddJq = ddJ[ngb_indices[k]]
        w[k] = np.minimum(ddJ, ddJq) * aPar.mu / distance
        w[k][edge_ngb[k]] = 0
    
    # Prepare data fidelity costs
    OP = (1-np.cos(2*np.pi*(np.arange(aPar.nB0)-offresCenter)/aPar.nB0)) / 2 * offresPenalty
    D = np.array([J[A, vxls] + OP[A], J[B, vxls] + OP[B]])
    print('DONE')

    # QPBO
    graphcut = aPar.graphcutLevel <= 0
    if graphcut:
        Vs = np.zeros((4, *ngb_indices.shape), dtype=float)
        for q in range(num_ngb):
            Vs[:, q, :] = np.array(w[q] * [
                V[abs(A - A[ngb_indices[q]])],
                V[abs(A - B[ngb_indices[q]])],
                V[abs(B - A[ngb_indices[q]])],
                V[abs(B - B[ngb_indices[q]])]])

        print('Solving MRF using QPBO...', end='')
        label = QPBO(D, Vs, ngb_indices)
        print('DONE')

        dB0[label == 0] = A[label == 0]
        dB0[label == 1] = B[label == 1]

    # ICM
    if aPar.nICMiter > 0:
        print('Solving MRF using ICM...', end='')
        dB0 = ICM(dB0, aPar.nB0, aPar.maxICMupdate, aPar.nICMiter, J, V, w, ngb_indices)
        print('DONE')
    return dB0


# Calculate LS error J as function of R2*
def get_R2_residuals(Y, dB0, C, nB0, nR2):
    dB0_wrapped = dB0 % nB0
    J = np.zeros(shape=(nR2, *Y.shape[1:]))
    for b in range(nB0):
        for r in range(nR2):
            y = Y[:, dB0_wrapped == b]
            J[r, dB0_wrapped == b] = np.linalg.norm(np.tensordot(C[r][b], y, axes=(1,0)), axis=0)**2
    return J


# Calculate LS error J as function of B0
def get_B0_residuals(Y, C, nB0, iR2cand, scale=1):
    num_voxels = Y.shape[1]
    J = np.zeros(shape=(nB0, num_voxels, len(iR2cand)))
    y = Y
    for r in range(len(iR2cand)):
        for b in range(nB0):
            J[b, :, r] = np.linalg.norm(np.tensordot(C[iR2cand[r]][b], y*scale, axes=(1,0)), axis=0)**2
    J = np.min(J, axis=-1) # minimum over R2* candidates
    return J


def modulation_vectors(nB0, N):
    B = np.zeros((nB0, N, N), dtype=complex)
    b = np.arange(nB0)[:, None]
    omega = 2.0 * np.pi * b / nB0
    n = np.arange(N)[None, :]
    B[b, n, n] = np.exp(1j * n * omega)
    return B


# Construct matrix RA
def modelMatrix(dPar, mPar, R2):
    RA = np.zeros(shape=(dPar.N, mPar.M), dtype=complex)
    for n in range(dPar.N):
        t = dPar.t1 + n * dPar.dt
        for m in range(mPar.M): # Loop over components/species
            for p in range(mPar.P):  # Loop over all resonances
                # Chemical shift between water and peak m (in ppm)
                omega = 2. * np.pi * GYRO * dPar.B0 * (mPar.CS[p] - mPar.CS[0])
                RA[n, m] += mPar.alpha[m][p]*np.exp(complex(-(t-dPar.t1)*R2, t*omega))
    return RA


# Get mean square signal magnitude within foreground
def getMeanEnergy(Y):
    energy = np.linalg.norm(Y, axis=0)**2
    thres = threshold_otsu(energy)
    return np.mean(energy[energy >= thres])


# Perform the actual reconstruction
def core_fatwater_separation(dPar, aPar, mPar, B0map=None, R2map=None):

    if mPar.realEstimates:
        raise NotImplementedError('Real-valued estimates not yet implemented.')

    Y = dPar.data.reshape(dPar.N, -1)
    shape = dPar.data.shape[1:]

    # Prepare matrices
    # Off-resonance modulation vectors, shape: (nB0, N, N)
    B = modulation_vectors(aPar.nB0, dPar.N)
    Bh = B.conj()
    RA, RAp, C, Qp = [], [], [], []
    
    for r in range(aPar.nR2):
        R2 = r*aPar.R2step
        RA.append(modelMatrix(dPar, mPar, R2))
        RAp.append(np.linalg.pinv(RA[r]))

    for r in range(aPar.nR2):
        C.append([])
        Qp.append([])
        # Null space projection matrix
        proj = np.eye(dPar.N)-np.dot(RA[r], RAp[r])
        for b in range(aPar.nB0):
            C[r].append(np.dot(np.dot(B[b], proj), Bh[b,:,:]))
            Qp[r].append(np.dot(RAp[r], Bh[b,:,:]))

    # For B0 index -> off-resonance in ppm
    B0step = 1.0/aPar.nB0/np.abs(dPar.dt)/GYRO/dPar.B0
    if aPar.algorithm != 'pass':
        V = []  # Precalculate discontinuity costs
        for b in range(aPar.nB0):
            V.append(min(b**2, (b-aPar.nB0)**2))
        V = np.array(V)

        scale = 1 / np.linalg.norm(Y)**2 # To avoid overflow
        J = get_B0_residuals(Y.reshape(dPar.N, -1), C, aPar.nB0, aPar.iR2cand, scale)
        offresPenalty = aPar.offresPenalty
        if aPar.offresPenalty > 0:
            offresPenalty *= getMeanEnergy(Y * scale)

        dB0 = calculate_fieldmap(J, V, aPar, shape, dPar.voxelsize, dPar.cyclic, offresPenalty, int(dPar.offresCenter/B0step))
    elif B0map is None:
        dB0 = np.zeros(np.prod(shape), dtype=int)
    else:
        dB0 = np.array(B0map.flatten()/B0step, dtype=int)

    if R2map is not None:
        R2 = np.array(R2map.flatten()/aPar.R2step, dtype=int)
    elif (aPar.nR2 > 1):
        J = get_R2_residuals(Y, dB0, C, aPar.nB0, aPar.nR2)
        R2 = np.argmin(J, axis=0) # brute force minimization
    else:
        R2 = np.zeros(np.prod(shape), dtype=int)

    # Find least squares solution given dB0 and R2
    rho = np.zeros(shape=(mPar.M, np.prod(shape)), dtype=complex)
    for r in range(aPar.nR2):
        for b in range(aPar.nB0):
            vxls = ((dB0 % aPar.nB0) == b) * (R2 == r)
            y = Y[:, vxls]
            rho[:, vxls] = np.dot(Qp[r][b], y)

    rho = rho.reshape(mPar.M, *shape)
    B0map = dB0.reshape(shape) * B0step
    R2map = R2.reshape(shape) * aPar.R2step

    return rho, B0map, R2map
