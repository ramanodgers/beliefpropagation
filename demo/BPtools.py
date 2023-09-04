import numpy as np 

def gaussianElimination(matrix, columns=None, diagonalize=True,
                        successfulCols=None, q=2):
        """
        gaussianElimination(matrix, columns=None, diagonalize=True, successfulCols=None, q=2)

        The Gaussian elimination algorithm in :math:`\mathbb F_q` arithmetics, turning a given
        matrix into reduced row echelon form by elementary row operations.

        .. warning:: This algorithm operates **in-place**!

        Parameters
        ----------
        matrix : np.int_t[:,::1]
            The matrix to operate on.
        columns : np.intp_t[:], optional
            A sequence of column indices, giving the the order in which columns of the matrix are
            visited. Defaults to ``range(matrix.shape[1])``.
        diagonalize : bool, True
            If ``True``, matrix elements above the pivots will be turned to zeros, leading to a
            diagonal submatrix. Otherwise, the result contains an upper triangular submatrix.
        successfulCols : np.intp_t[::1], optinonal
            Numpy array in which successfully diagonalized column indices are stored. If supplied,
            this array will be used for the return value. Otherwise, a new array will be created,
            resulting in a slight performance drain.
        q : int, optional
            Field size in which operations should be performed. Defaults to ``2``.

        Returns
        -------
        np.intp_t[::1]
            Indices of successfully diagonalized columns.
        """
        nrows = matrix.shape[0]
        ncols = matrix.shape[1]
        curRow = 0
        colIndex = 0
        numSuccessfulCols = 0
        # assert q < cachedInvs.shape[0]

        if successfulCols is None:
            successfulCols = np.empty(nrows, dtype=np.intp)
        if columns is None:
            columns = np.arange(ncols, dtype=np.intp)
        while True:
            if colIndex >= columns.shape[0]:
                break
            curCol = columns[colIndex]
            # search for a pivot row
            pivotRow = -1
            for row in range(curRow, nrows):
                val = matrix[row, curCol]
                if val != 0:
                    pivotRow = row
                    break
            if pivotRow == -1:
                # did not find a pivot row -> this column is linearly dependent of the previously
                # visited; continue with next column
                colIndex += 1
                continue
            if pivotRow > curRow:
                # swap rows
                for i in range(ncols):
                    val = matrix[curRow, i]
                    matrix[curRow, i] = matrix[pivotRow, i]
                    matrix[pivotRow, i] = val
            # do the actual pivoting
            if matrix[curRow, curCol] > 1:
                # "divide" by pivot element to set it to 1
                if q > 2:
                    factor = cachedInvs[q, matrix[curRow, curCol]]
                    for i in range(ncols):
                        matrix[curRow, i] = (matrix[curRow, i] * factor) % q
            for row in range(curRow + 1, nrows):
                val = matrix[row, curCol]
                if val != 0:
                    for i in range(ncols):
                        if q == 2:
                            matrix[row, i] ^= matrix[curRow, i]
                        else:
                            matrix[row, i] =  (matrix[row, i] -val*matrix[curRow, i]) % q
            successfulCols[numSuccessfulCols] = curCol
            numSuccessfulCols += 1
            if numSuccessfulCols == nrows:
                break
            curRow += 1
            colIndex += 1
        if diagonalize:
            for colIndex in range(numSuccessfulCols):
                curCol = successfulCols[colIndex]
                for row in range(colIndex):
                    val = matrix[row, curCol]
                    if val != 0:
                        for i in range(ncols):
                            if q == 2:
                                matrix[row, i] ^= matrix[colIndex, i]
                            else:
                                matrix[row, i] = (matrix[row, i] - val*matrix[colIndex, i]) % q
        return successfulCols[:numSuccessfulCols]

def rank(matrix, q=2):
    """Return the rank (in GF(q)) of a matrix."""
    diagCols = gaussianElimination(matrix.copy(), diagonalize=False, q=q)
    return diagCols.size

def orthogonalComplement(matrix, columns=None, q=2):
    """Computes an orthogonal complement (in GF(q)) to the given matrix."""
    matrix = np.asarray(matrix.copy())
    m, n = matrix.shape
    unitCols = gaussianElimination(matrix, columns, diagonalize=True, q=q)
    nonunitCols = np.array([x for x in range(n) if x not in unitCols])
    rank = unitCols.size
    nonunitPart = matrix[:rank, nonunitCols].transpose()
    k = n - rank
    result = np.zeros((k, n), dtype=np.int32)
    for i, c in enumerate(unitCols):
        result[:, c] = (-nonunitPart[:, i]) % q
    for i, c in enumerate(nonunitCols):
        result[i, c] = 1
    return result


def convert(strings, leng, plural=True, alpha=True):
    """given a string of stabilizers and number of encoding qubits, convert into 2d array parity check matrix"""
    if plural:
        total = len(strings)
    else:
        total = 1
    stabilizers = np.zeros([total,leng],dtype=int)
    filler=[]
    if alpha:
        match_x = 'X'
        match_z = 'Z'
        #match_y = 'Y'
    else:
        match_x = '1'
        match_z = '1'
    length = leng
    stabilizers =np.zeros([total,leng],dtype=int)
    for i in range(total):
        curr_stab = np.zeros((length,),dtype=int)
        for j in range(leng):
            if plural:
                if strings[i][j]==match_x:
                    curr_stab[j]=1
                elif strings[i][j]==match_z:
                    curr_stab[j]=1
                else:
                    curr_stab[j]=0
            else:
                if strings[j]==match_x:
                    curr_stab[j]=1
                elif strings[j]==match_z:
                    curr_stab[j]=1
                else:
                    curr_stab[j]=0
        stabilizers[i]=curr_stab
    return stabilizers  

def generator(H):
    #returns the corresponding generator matrix for a given H PCM
     return np.transpose(orthogonalComplement(H))

def AIk(M):
    #returns the given matrix in the A:Ik form 
    return np.concatenate((M,np.identity(M.shape[0], dtype = np.int64)), axis = 1)

def word_gen(G, x = None):
    #generates some codeword for a given generator matrix 

    rng = np.random.default_rng()
    #x is the encoded info (k x 1)
    if x is None:
        x = rng.integers(2, size=G.shape[1])
    # y is the codeword
    y = np.remainder(np.dot(G,x),2)
    return y 

def random_data(length):
    rng = np.random.default_rng()
    return rng.integers(2, size = length)
    
def solver(m, b):
    #matrix inversion solver
    b = b.reshape(-1,1)
    aug = np.hstack([m, b])
    successful = gaussianElimination(aug)
    xguess = np.zeros(m.shape[1],dtype = int)
    i = 0
    while i < len(xguess):
        xguess[successful[i]] = aug[i][aug.shape[1]-1]
        i +=1
    return xguess

def iid_error(codeword, p):
    # randomly disturbs a given codeword with iid probability p of an error on any bit
    # generate a random codeword length string and add it to the codeword mod 2
    rng = np.random.default_rng()
    error = rng.choice(2, size = codeword.shape[0], p = [1-p,p])
    corrupted = np.remainder(codeword + error, 2) 
    return corrupted

def dist_error(codeword, distance):
    #corrupts the data with a given number of bitflips.
    error = np.array([0] * (codeword.shape[0]-distance) + [1] * (distance))
    np.random.shuffle(error)
    corrupted = np.remainder(codeword + error, 2, dtype= int) 
    return corrupted

def code_rate(M):
    # NB code rate assumes a linear code for quantum encoding, usually not linear code
    m = M.shape[0]
    n = M.shape[1]
    k = n-m
    return "Code Rate:" + str(k/n)

def HGP(h1,h2):
    #generates a hypergraph product code for quantum decoders 
    m1,n1=h1.shape
    m2,n2=h2.shape
    m1=np.identity(m1,dtype=int)
    n1=np.identity(n1,dtype=int)
    m2=np.identity(m2,dtype=int)
    n2=np.identity(n2,dtype=int)
    
    hx1=np.kron(h1,n2)
    hx2=np.kron(m1,h2.T)
    hx = np.hstack([hx1,hx2])
    hz1=np.kron(n1,h2)
    hz2=np.kron(h1.T,m2)
    hz = np.hstack([hz1,hz2 ])
    
    hxtemp=np.vstack([np.zeros(hz.shape,dtype=int),hx])
    hztemp=np.vstack([hz,np.zeros(hx.shape,dtype=int)])
    return np.hstack([hztemp,hxtemp])


# input is log ratio guesses as in the decoder self.Q form
# output colz are the first columns in bp_sort that can be reduced using gaussianElimination
# second output is the binary string for which of those columns have errors by OSD
def osd_0(H, syndrome, bp_probs):
    bp_sort = np.argsort(bp_probs)
    colz  = gaussianElimination(H.copy(), columns=bp_sort, diagonalize=False)
    H_OSD = H[:,colz]
    # compute inverse
    H_AUG = np.ascontiguousarray(np.hstack([H_OSD, syndrome[:,np.newaxis]]), dtype = int)
    gaussianElimination(H_AUG)
    #OSD does the same thing as solver()
    return colz, H_AUG[:H_OSD.shape[1], -1]


def decimalToBinary(n):
    return bin(n).replace("0b", "")

def format_input(binstr, length):
    short = np.array([*binstr], dtype=np.int8)
    if len(short) <= length: 
        return np.pad(short, (length - len(short), 0), 'constant')
    else:
        raise ValueError("input too large")
        
def arr2string(arr):
    return ''.join(str(int(x)) for x in arr)

    
class syndrome_BP:
    def __init__(self, H, syndrome, max_iter, p, OSD = False, higher = False):
        # this class assumes the iid errors
        # syndrome based belief propagation
        self.syndrome = np.array(syndrome)
        self.H = H
        self.Q = np.zeros(self.H.shape[1])
        self.m = self.H.shape[0]
        self.n = self.H.shape[1]
        self.k = self.n - self.m
        # if max_iter < H.shape[1]:
        #     self.max_iter = H.shape[1]
        # else:
        self.max_iter = max_iter
        self.guess = np.zeros(self.H.shape[1], dtype = int)
        self.p = p
        self.OSD  = OSD
        self.higherOSD  = higher

        #the log probability that the noise values for each bit is zero 
        self.Lconst = np.full((self.H.shape[1]),np.log(1-self.p) - np.log(self.p))
        
        #belief prop
        self.Lqij = np.zeros((self.H.shape[0],self.H.shape[1]))
        self.Lrij = np.zeros((self.m,self.n))
        self.coords = []
        
    def phi(self,x):
        if x ==0:
            return 0
        return -np.log(np.tanh(np.abs(x)/2))
    
    def r_update(self):
            
        for coord in self.coords:
            if self.syndrome[coord[0]] == 1:
                sign = -1
            else:
                sign = 1
            tempsum = 0
            for subcoord in self.coords:
                if subcoord[0] == coord[0]:
                    if subcoord[1] != coord[1]:
                        if self.Lqij[subcoord[0]][subcoord[1]] == 0:
                            continue
                        else:
                            temp = self.Lqij[subcoord[0]][subcoord[1]]
                            tempsum += self.phi(temp)
                        
            self.Lrij[coord[0]][coord[1]] = sign * self.phi(tempsum)


    def q_update(self):
        for coord in self.coords: 
            tempsum = 0
            for subcoord in self.coords:
                if subcoord[1] == coord[1]:
                    if subcoord[0] != coord[0]:
                        tempsum += self.Lrij[subcoord[0]][subcoord[1]]
            
            self.Lqij[coord[0]][coord[1]] = self.Lconst[coord[1]] + tempsum

    def guesser(self):
        addend = np.sum(self.Lrij, axis=0)
        Q  = self.Lconst + addend
        self.change = np.sum(np.abs(Q - self.Q))/len(Q)
        self.Q = Q
        for var in range(self.n):

            if self.Q[var] < 0:
                self.guess[var] = 1
            else:
                self.guess[var] = 0


        if np.any(self.guess):
            He = np.remainder(np.dot(self.H,self.guess),2)
            #stop condition
            if np.array_equal(He,self.syndrome):
                return True
            else:
                return False
    
    def initialize(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.H[i][j] == 1:
                    self.coords.append((i,j))
        for coord in self.coords: 
            self.Lqij[coord[0]][coord[1]] = 1
            self.Lrij[coord[0]][coord[1]] = 1
        

    def decoder(self):
        if len(self.syndrome) != self.m:
            raise ValueError("incorrect block size")
        self.initialize()
        
        #immediately returns if the syndrome is zero 
        if not np.any(self.syndrome):
            return self.guess, None
        
        for _ in range(self.max_iter):
            self.r_update()

            self.q_update()
            
            #here we only do OSD if the result does not converge
            result = self.guesser()
            if result:
                #success
                return self.guess, False
        
        if self.OSD or self.higherOSD:
            self.guess = np.zeros(self.H.shape[1], dtype = int)

            cols, OSDerror = osd_0(self.H, self.syndrome, self.Q)
            for i in range(len(cols)):
                self.guess[cols[i]] = OSDerror[i]
            He = np.remainder(np.dot(self.H,self.guess),2)
            # stop condition
            if not np.array_equal(He,self.syndrome):
                print("\n what \n")
                
        tempweight = np.sum(self.guess) 
        
        
        #now for higher-order OSD
        if self.higherOSD:
            oldes = self.guess
            lamb = 10
            self.guess = np.zeros(self.H.shape[1], dtype = int)
            Hs= self.H[:,cols] #correct
            first_term  = OSDerror
            
            first_guess =self.guess
            tcolz = [*range(self.H.shape[1])]
            tcolz = np.setdiff1d(tcolz,cols)
            bp_sort = np.argsort(self.Q)
            t_sort = bp_sort[np.in1d(bp_sort, tcolz)]
            Ht= self.H[:,t_sort]
            
            #weight-one configs
            for i, bit in enumerate(t_sort):
                self.guess = np.zeros(self.H.shape[1], dtype = int)
                et = np.zeros(len(t_sort), dtype = int)
                et[i] = 1
                temp = np.remainder(np.dot(Ht,et),2)
                second_term  = solver(Hs,temp)
                first_half = np.remainder(first_term + second_term,2)
                # print(first_half.shape)
                for i in range(len(cols)):
                    self.guess[cols[i]] = first_half[i]
                self.guess[bit] = 1
                weight = np.sum(first_half) + 1
                if weight < tempweight:
                    tempweight = weight
                    oldes = self.guess
                    return self.guess, True

            # weight two configs
            # for combo in combinations(t_sort[:lamb],2):
            #     self.guess = np.zeros(self.H.shape[1], dtype = int)
            #     et = np.zeros(len(t_sort), dtype = int)
            #     et[np.where(t_sort == combo[0])] = 1
            #     et[np.where(t_sort == combo[1])] = 1
            #     second_term  = solver(Hs,np.remainder(np.dot(Ht,et),2))
            #     first_half = np.remainder(first_term + second_term,2)
            #     for i in range(len(cols)):
            #         self.guess[cols[i]] = first_half[i]
            #     self.guess[combo[0]] = 1
            #     self.guess[combo[1]] = 1
            #     weight = np.sum(first_half) + 2
            #     if weight < tempweight:
            #         tempweight = weight
            #         oldes = self.guess
            #         print("higher worked")
            #         return self.guess, True
            self.guess = oldes
   

        return self.guess, True  
