import numpy as np
import random
import matplotlib.pyplot as plt

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def mutasi(x, sigma):
    return x + sigma * np.random.randn(*x.shape)

def update_sigma(sigma):
    return sigma * np.exp(np.random.randn(*sigma.shape))

def rekombinasi(parents):
    return np.mean(parents, axis=0)

def ES_plus(miu, lambda_, n_max_iter):
    n = 2
    parents = np.random.uniform(-5, 5, (miu, n))
    sigmas = np.random.uniform(0, 1, (miu, n))
    iterasiku = 0
    
    # Plot sebaran populasi awal generasi 
    plt.figure(1)
    globalMinima = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584458, -1.848126]]
    plt.scatter(*zip(*globalMinima), marker='X', color='red', zorder=1)
    plt.scatter(*zip(*parents))
    plt.title('Distribusi Populasi Awal Generasi')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    fitness_values = []
    
    for i in range(n_max_iter):
        offspring = []
        new_sigmas = []
        
        for j in range(lambda_):
            index_parent = np.random.choice(miu, 2, replace=False)
            selected_parents = parents[index_parent]
            hasil_rekombinasi = rekombinasi(selected_parents)
            sigma = sigmas[j % miu]
            hasil_mutasi = mutasi(hasil_rekombinasi, sigma)
            offspring.append(hasil_mutasi)
            new_sigma = update_sigma(sigma)
            new_sigmas.append(new_sigma)
        
        offspring = np.vstack(offspring)
        parents = np.vstack([parents, offspring])
        sigmas = np.vstack([sigmas, new_sigmas])
        fitness = np.apply_along_axis(himmelblau, 1, parents)
        index_terpilih = np.argsort(fitness)[:miu]
        parents = parents[index_terpilih]
        sigmas = sigmas[index_terpilih]
        fitness_values.append(np.min(fitness))
        if np.min(fitness) == 0:
            iterasiku = i
            break
    
    print(f"Generasi Atau Iterasi Berhasil berhenti disaat iterasi mencari gen ke-{iterasiku}")
    print(f"Karena nilai fitness atau F(x,y) sudah mencapai == {np.min(fitness)}")
    # Plot sebaran populasi akhir generasi  
    plt.figure(2)
    globalMinima = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584458, -1.848126]]
    plt.scatter(*zip(*globalMinima), marker='X', color='red', zorder=1)
    plt.scatter(*zip(*parents))
    plt.title('Distribusi Populasi Akhir Generasi')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Plot fitness over time
    plt.figure(3)
    plt.plot(fitness_values)
    plt.title('Fitness over Time')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    
    plt.show()
    
    return parents[np.argmin(np.apply_along_axis(himmelblau, 1, parents))]

# Set the seed for the random number generator
SEED_RANDOM = 42
random.seed(SEED_RANDOM)
np.random.seed(SEED_RANDOM)

miu = 100
lambda_ = 7 * miu
n_max_iter = 500

best_individual = ES_plus(miu, lambda_, n_max_iter)
best_x,best_y = best_individual
print(f"Nilai Terbaik yang dihasilkan untuk variabel x = {best_x}, sedangkan untuk variabel y = {best_y}")
print(f"F(x,y) Himmelblau : {himmelblau(best_individual)}")
