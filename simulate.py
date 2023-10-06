import environment

if __name__ == '__main__':

    env = environment.Environment(population_size = 1000, 
                                  number_of_countries = 6, 
                                  number_of_transactions = 100, 
                                  alpha = 4, 
                                  beta = 0.5,
                                  gamma = 2,
                                  delta = 2, 
                                  epsilon = 0.5)

    for i in range(1000):
        env.one_episode()
        print("completed episode numer", i + 1)

    env.show_history(save=True)