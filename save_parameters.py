import numpy as np

N = [ 10**3, 5*10**3, 10**4 ]

def main():

    for n in N:
        if n == 10**3: x = 9e-3
        else: x = 5e-3
        # define probabilities of interest
        prob = [ 8e-1, 7e-1, 6e-1, 1e-1, 1e-2, x,                         # connected regine (high - low)
                1/(3*n) + (2*np.log(n))/(3*n), 2/(3*n) + np.log(n)/(3*n), # supercritical regime
                1/(2*n), 1/(10*n) ]                                       # subscritical regime
        print(n, prob)
        print()
        # save probabilities
        with open('data/out/keys{}.txt'.format(str(n)), 'w') as f:
            #saving keys to file
            f.write(str(list(prob)))
    # save the explored values of n
    np.save('data/out/sizes.npy', N)
    # save resistances thresholds
    np.save('data/out/res_phase1.npy', [ 0.25, 0.5, 0.75, 1 ])  # phase 1 values

if __name__ == "__main__":
    main()
