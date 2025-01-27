import numpy as np

N = [ 10**3, 5*10**3, 10**4 ]
N2 = np.arange(10**3, 10**4 + 10**3, 10**3)

def main():
    for n in N:
        if n == 10**3: x = 9e-3
        else: x = 5e-3
        # define probabilities of interest
        prob = [ 8e-1, 7e-1, 6e-1, 1e-1, 1e-2, x,                         # connected regime (high - low)
                1/(3*n) + (2*np.log(n))/(3*n), 2/(3*n) + np.log(n)/(3*n), # supercritical regime
                1/(2*n), 1/(10*n) ]                                       # subscritical regime
        prob2 = np.arange(1e-1 - 0.05, 1e-2, -(1e-1 - 1e-2)/20)
        prob3 = [ (1-k)/n + (k*np.log(n))/n for k in [1/6, 1/2, 5/6] ]
        print('prob', n, prob)
        print()
        print('prob2', n, prob2)
        print()
        print('prob3', n, prob2)
        print()

        # save probabilities
        with open('data/keys{}.txt'.format(str(n)), 'w') as f:
            #saving keys to file
            f.write(str(list(prob)))

        with open('data/keys_ref.txt'.format(str(n)), 'w') as f:
            #saving keys to file
            f.write(str(list(prob2)))

        with open('data/keys_gc_{}.txt'.format(str(n)), 'w') as f:
            #saving keys to file
            f.write(str(list(prob3)))

    prob.extend(prob2)
    prob.sort(reverse=True)
    print('jointed', prob, len(prob), end= '\n\n')

    for n in N2:
        if n == 10**3: x = 9e-3
        else: x = 5e-3
        prob = [ 8e-1, 7e-1, 6e-1, 1e-1, 1e-2, x,                         # connected regime (high - low)
                1/(3*n) + (2*np.log(n))/(3*n), 2/(3*n) + np.log(n)/(3*n), # supercritical regime
                1/(2*n), 1/(10*n) ]                                       # subscritical regime

        with open('data/keys{}.txt'.format(str(n)), 'w') as f:
            #saving keys to file
            f.write(str(list(prob)))

    # save the explored values of n
    np.save('data/sizes.npy', N)
    # save the explored values of n
    np.save('data/sizes_ref.npy', N2)
    # save resistances thresholds
    np.save('data/res_phase1.npy', [ 0.25, 0.5, 0.75, 1 ])  # phase 1 values
    np.save('data/res_phase2.npy', [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ])  # phase 2 values
    np.save('data/res_phase3.npy', [ 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])  # phase 3 values
    np.save('data/res_phase4.npy', [ 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625, 0.675])  # phase 4 values

    # suitable pred resistances for phase 2 (inflection point)
    np.save('data/res_pred_0.8', [ 0.65, 0.7, 0.8, 0.85, 0.9] )
    np.save('data/res_pred_0.7', [ 0.45, 0.55, 0.6, 0.65, 0.7] )
    np.save('data/res_pred_0.6', [ 0.35, 0.4, 0.45, 0.55, 0.6] )


if __name__ == "__main__":
    main()
