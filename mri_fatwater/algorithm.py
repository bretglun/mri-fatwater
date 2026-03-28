import thinqpbo as tq
import numpy as np
from skimage.filters import threshold_otsu
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


# Find all local minima of discretely evaluated function f(t) with period T
def findMinima(f): return np.where((f < np.roll(f, 1))*(f < np.roll(f, -1)))[0]


# In each voxel, find two smallest local residual minima in a period of omega
def findTwoSmallestMinima(J):
    A = np.zeros(J.shape[1], dtype=int)
    B = np.zeros(J.shape[1], dtype=int)
    for q in range(J.shape[1]):
        minima = sorted(findMinima(J[:, q]), key=lambda b: J[b, q])
        if len(minima) >= 2:
            A[q], B[q] = minima[:2]
        elif len(minima) == 1:
            A[q] = B[q] = minima[0]
    return A, B


# 2D measure of isotropy defined as
# the square area over the square perimeter (area normalized to 1)
def isotropy2D(dx, dy): return np.sqrt(dx*dy)/(2*(dx+dy))


# 3D measure of isotropy defined as
# the cube volume over the cube area (volume normalized to 1)
def isotropy3D(dx, dy, dz): return (dx*dy*dz)**(2/3)/(2*(dx*dy+dx*dz+dy*dz))


def getHigherLevel(level):
    high = {'L': level['L'] + 1}
    # Isotropy promoting downsampling
    maxIsotropy = 0
    for sx in [1, 2]:
        for sy in [1, 2]:
            for sz in [1, 2]:  # Loop over all 2^3=8 downscaling combinations
                # at least one dimension must change and the size of all
                # dimensions at lower level must permit any downscaling
                if (sx*sy*sz > 1 and level['shape'][0] >= sx and
                   level['shape'][1] >= sy and level['shape'][2] >= sz):
                    if (level['shape'][0] == 1):
                        iso = isotropy2D(level['voxelsize'][1]*sy, level['voxelsize'][2]*sz)
                    elif (level['shape'][1] == 1):
                        iso = isotropy2D(level['voxelsize'][0]*sx, level['voxelsize'][2]*sz)
                    elif (level['shape'][2] == 1):
                        iso = isotropy2D(level['voxelsize'][0]*sx, level['voxelsize'][1]*sy)
                    else:
                        iso = isotropy3D(
                          level['voxelsize'][0]*sx, level['voxelsize'][1]*sy, level['voxelsize'][2]*sz)
                    if iso > maxIsotropy:
                        maxIsotropy = iso
                        high['block'] = (sx, sy, sz)
    high['voxelsize'] = tuple(vs * s for vs, s in zip(level['voxelsize'], high['block']))
    high['shape'] = tuple(int(np.ceil(n/s)) for n, s in zip(level['shape'], high['block']))
    return high


def getHighLevelResidualImage(J, high, level):
    J = J.reshape(J.shape[0], *level['shape'])
    Jhigh = np.zeros((J.shape[0], level['shape'][0] + level['shape'][0] % high['block'][0],
                                  level['shape'][1] + level['shape'][1] % high['block'][1],
                                  level['shape'][2] + level['shape'][2] % high['block'][2]))
    Jhigh[:, :level['shape'][0], :level['shape'][1], :level['shape'][2]] = J
    Jhigh = Jhigh.reshape((J.shape[0], high['shape'][0], high['block'][0], high['shape'][1], high['block'][1], high['shape'][2], high['block'][2])).mean(axis=(2,4,6))
    return Jhigh.reshape(Jhigh.shape[0], -1)


def getB0fromHighLevel(dB0high, level, high):
    return np.repeat(np.repeat(np.repeat(dB0high.reshape(high['shape']), high['block'][2], axis=2), high['block'][1], axis=1), high['block'][0], axis=0)[:level['shape'][0], :level['shape'][1], :level['shape'][2]].flatten()


def calculateFieldMap(nB0, level, graphcutLevel, multiScale, maxICMupdate,
                      nICMiter, J, V, mu, offresPenalty=0, offresCenter=0):
    A, B = findTwoSmallestMinima(J)
    dB0 = np.array(A)

    # Multiscale recursion
    if dB0.size == 1:  # Trivial case at coarsest level with only one voxel
        print(f'Level {level['shape']}: Trivial case')
        return dB0

    if multiScale:
        high = getHigherLevel(level)
        Jhigh = getHighLevelResidualImage(J, high, level)
        # Recursion:
        dB0high = calculateFieldMap(nB0, high, graphcutLevel, multiScale,
                                    maxICMupdate, nICMiter, Jhigh, V, mu,
                                    offresPenalty, offresCenter)
        dB0 = getB0fromHighLevel(dB0high, level, high)
        print(f'Level {level['shape']}: ')

    # Prepare MRF
    print('Preparing MRF...', end='')
    # Prepare discontinuity costs
    neighbourhood = np.array([
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1]
    ])

    cyclic = [False] * len(level['shape'])

    ngb_indices, edge_ngb = get_neighbour_indices(neighbourhood, level['shape'], cyclic)
    num_ngb = ngb_indices.shape[0]
    
    # 2nd derivative of residual function
    # NOTE: No division by square(steplength) since
    # square(steplength) not included in V
    vxls = range(np.prod(level['shape']))
    ddJ = (J[(A+1) % nB0, vxls] + J[(A-1) % nB0, vxls] - 2 * J[A, vxls])

    w = np.zeros((ngb_indices.shape))
    for k, ngb in enumerate(neighbourhood):
        distance = np.linalg.norm(ngb * level['voxelsize'])
        ddJq = ddJ[ngb_indices[k]]
        w[k] = np.minimum(ddJ, ddJq) * mu / distance
        w[k][edge_ngb[k]] = 0
    
    # Prepare data fidelity costs
    OP = (1-np.cos(2*np.pi*(np.arange(nB0)-offresCenter)/nB0)) / 2 * offresPenalty
    D = np.array([J[A, vxls] + OP[A], J[B, vxls] + OP[B]])
    print('DONE')

    # QPBO
    graphcut = level['L'] >= graphcutLevel
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
    if nICMiter > 0:
        print('Solving MRF using ICM...', end='')
        dB0 = ICM(dB0, nB0, maxICMupdate, nICMiter, J, V, w, ngb_indices)
        print('DONE')
    return dB0


# Calculate initial phase phi according to
# Bydder et al. MRI 29 (2011): 216-221.
def getPhi(Y, D):
    phi = np.zeros((Y.shape[1]))
    for i in range(Y.shape[1]):
        y = Y[:, i]
        phi[i] = .5*np.angle(np.dot(np.dot(y.transpose(), D), y))
    return phi


# Calculate phi, remove it from Y and return separate real and imag parts
def getRealDemodulated(Y, D):
    phi = getPhi(Y, D)
    y = Y/np.exp(1j*phi)
    return np.concatenate((np.real(y), np.imag(y))), phi


# Calculate LS error J as function of R2*
def get_R2_residuals(Y, dB0, C, nB0, nR2, D=None):
    J = np.zeros(shape=(nR2, *Y.shape[1:]))
    for b in range(nB0):
        for r in range(nR2):
            if not D:  # complex-valued estimates
                y = Y[:, dB0 == b]
            else:  # real-valued estimates
                y = getRealDemodulated(Y[:, dB0 == b], D[r][b])[0]
            J[r, dB0 == b] = np.linalg.norm(np.tensordot(C[r][b], y, axes=(1,0)), axis=0)**2
    return J


# Calculate LS error J as function of B0
def get_B0_residuals(Y, C, nB0, iR2cand, D=None, scale=1):
    num_voxels = Y.shape[1]
    J = np.zeros(shape=(nB0, num_voxels, len(iR2cand)))
    if not D: # complex-valued estimates
        y = Y
    for r in range(len(iR2cand)):
        for b in range(nB0):
            if D:  # real-valued estimates
                y = getRealDemodulated(Y, D[r][b])[0]
            J[b, :, r] = np.linalg.norm(np.tensordot(C[iR2cand[r]][b], y*scale, axes=(1,0)), axis=0)**2
    J = np.min(J, axis=-1) # minimum over R2* candidates
    return J


# Construct modulation vectors for each B0 value
def modulationVectors(nB0, N):
    B, Bh = [], []
    for b in range(nB0):
        omega = 2.*np.pi*b/nB0
        B.append(np.eye(N)+0j*np.eye(N))
        for n in range(N):
            B[b][n, n] = np.exp(complex(0., n*omega))
        Bh.append(B[b].conj())
    return B, Bh


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


# Get matrix Dtmp defined so that D = Bconj*Dtmp*Bh
# Following Bydder et al. MRI 29 (2011): 216-221.
def getDtmp(A):
    Ah = A.conj().T
    inv = np.linalg.inv(np.real(np.dot(Ah, A)))
    Dtmp = np.dot(A.conj(), np.dot(inv, Ah))
    return Dtmp


# Separate and concatenate real and imag parts of complex matrix M
def realify(M):
    re = np.real(M)
    im = np.imag(M)
    return np.concatenate((np.concatenate((re, im)), np.concatenate((-im, re))), 1)


# Get mean square signal magnitude within foreground
def getMeanEnergy(Y):
    energy = np.linalg.norm(Y, axis=0)**2
    thres = threshold_otsu(energy)
    return np.mean(energy[energy >= thres])


# Perform the actual reconstruction
def core_fatwater_separation(dPar, aPar, mPar, B0map=None, R2map=None):
    determineB0 = aPar.graphcutLevel is not None or aPar.nICMiter > 0
    determineR2 = (aPar.nR2 > 1) and (R2map is None)

    Y = dPar.data.reshape(dPar.N, -1)
    shape = dPar.data.shape[1:]

    # Prepare matrices
    # Off-resonance modulation vectors (one for each off-resonance value)
    B, Bh = modulationVectors(aPar.nB0, dPar.N)
    RA, RAp, C, Qp = [], [], [], []
    D = None
    if mPar.realEstimates:
        D = []  # Matrix for calculating phi (needed for real-valued estimates)
    for r in range(aPar.nR2):
        R2 = r*aPar.R2step
        RA.append(modelMatrix(dPar, mPar, R2))
        if mPar.realEstimates:
            D.append([])
            Dtmp = getDtmp(RA[r])
            for b in range(aPar.nB0):
                D[r].append(np.dot(B[b].conj(), np.dot(Dtmp, Bh[b])))
            RA[r] = np.concatenate((np.real(RA[r]), np.imag(RA[r])))
        RAp.append(np.linalg.pinv(RA[r]))

    if mPar.realEstimates:
        for b in range(aPar.nB0):
            B[b] = realify(B[b])
            Bh[b] = realify(Bh[b])
    for r in range(aPar.nR2):
        C.append([])
        Qp.append([])
        # Null space projection matrix
        proj = np.eye(dPar.N*(1+mPar.realEstimates))-np.dot(RA[r], RAp[r])
        for b in range(aPar.nB0):
            C[r].append(np.dot(np.dot(B[b], proj), Bh[b]))
            Qp[r].append(np.dot(RAp[r], Bh[b]))

    # For B0 index -> off-resonance in ppm
    B0step = 1.0/aPar.nB0/np.abs(dPar.dt)/GYRO/dPar.B0
    if determineB0:
        V = []  # Precalculate discontinuity costs
        for b in range(aPar.nB0):
            V.append(min(b**2, (b-aPar.nB0)**2))
        V = np.array(V)

        level = {'L': 0, 'shape': shape, 'block': (1,)*len(dPar.voxelsize), 'voxelsize': dPar.voxelsize}
        scale = 1 / np.linalg.norm(Y)**2 # To avoid overflow
        J = get_B0_residuals(Y.reshape(dPar.N, -1), C, aPar.nB0, aPar.iR2cand, D, scale)
        offresPenalty = aPar.offresPenalty
        if aPar.offresPenalty > 0:
            offresPenalty *= getMeanEnergy(Y * scale)

        dB0 = calculateFieldMap(aPar.nB0, level, aPar.graphcutLevel,
                                aPar.multiScale, aPar.maxICMupdate,
                                aPar.nICMiter, J, V, aPar.mu,
                                offresPenalty, int(dPar.offresCenter/B0step))
    elif B0map is None:
        dB0 = np.zeros(np.prod(shape), dtype=int)
    else:
        dB0 = np.array(B0map.flatten()/B0step, dtype=int)

    if determineR2:
        J = get_R2_residuals(Y, dB0, C, aPar.nB0, aPar.nR2, D)
        R2 = np.argmin(J, axis=0) # brute force minimization
    elif R2map is None:
        R2 = np.zeros(np.prod(shape), dtype=int)
    else:
        R2 = np.array(R2map.flatten()/aPar.R2step, dtype=int)

    # Find least squares solution given dB0 and R2
    rho = np.zeros(shape=(mPar.M, np.prod(shape)), dtype=complex)
    for r in range(aPar.nR2):
        for b in range(aPar.nB0):
            vxls = (dB0 == b) * (R2 == r)
            if not D:  # complex estimates
                y = Y[:, vxls]
            else:  # real-valued estimates
                y, phi = getRealDemodulated(Y[:, vxls], D[r][b])
            rho[:, vxls] = np.dot(Qp[r][b], y)
            if D:
                #  Assert phi is the phase angle of water
                phi[rho[0, vxls] < 0] += np.pi
                rho[:, vxls] *= np.exp(1j*phi)

    rho = rho.reshape(mPar.M, *shape)
    B0map = dB0.reshape(shape) * B0step
    R2map = R2.reshape(shape) * aPar.R2step

    return rho, B0map, R2map
