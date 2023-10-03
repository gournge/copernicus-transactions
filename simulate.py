import environment

if __name__ == '__main__':

    env = environment.Environment(population_size=1000, number_of_countries=5, number_of_transactions=50, delta=10)

    for i in range(500):
        env.one_episode()
        print("completed episode numer", i + 1)

    env.show_history()