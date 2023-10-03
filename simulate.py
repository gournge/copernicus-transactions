import environment

if __name__ == '__main__':

    env = environment.Environment()

    for _ in range(100):

        env.one_episode()

    env.show_history()