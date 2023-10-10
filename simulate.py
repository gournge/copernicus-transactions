import environment
import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--population_size", type=int, default=1000)
    parser.add_argument("--number_of_countries", type=int, default=6)
    parser.add_argument("--number_of_currencies", type=int, default=4)
    parser.add_argument("--number_of_transactions", type=int, default=100)

    parser.add_argument("--increase_weakest_currency", type=str, default='True', choices=['True', 'False'], help='whether to increase the weakest currency over all the episodes')
    parser.add_argument("--even_countries_currency_spread", type=str, default='True', choices=['True', 'False'], help='whether there should be an evenly distributed spread of assigned currencies between countries')
    parser.add_argument("--verbose", type=str, default='True', choices=['True', 'False'], help='whether to log progress of the simulation')
    
    parser.add_argument("--alpha", type=float, default=2, help='how much more money is an agent expected to have in his home currency')
    parser.add_argument("--beta", type=float, default=0.5, help='what maximum percent of his budget is an agent willing to use for a transaction')
    parser.add_argument("--gamma", type=float, default=0.0086, help='how much more likely is an agent to make a transaction  with someone from their own country')
    parser.add_argument("--delta", type=float, default=1.5, help='how much more impact on the probability of choosing currencies does their value have')
    parser.add_argument("--epsilon", type=float, default=0.5, help='how much impact on the probability of choosing currencies does agent\'s wallet contents have')
    parser.add_argument("--zeta", type=float, default=0.5, help='how much impact on the probability of choosing currencies does agent\'s wallet contents have')

    parser.add_argument("--number_of_episodes", type=int, default=500)

    parser.add_argument("--save_figure", type=str, default='False', choices=['True', 'False'])
    parser.add_argument("--figure_convolution_window_size", type=int, default=10, help='with how many values do you average out curve')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = get_args()

    env = environment.Environment(population_size = args.population_size,
                                  number_of_countries = args.number_of_countries,
                                  number_of_currencies = args.number_of_currencies,
                                  number_of_transactions = args.number_of_transactions,
                                  number_of_episodes = args.number_of_episodes,
                                  increase_weakest_currency = (args.increase_weakest_currency == 'True'),
                                  even_countries_currency_spread = (args.even_countries_currency_spread == 'True'),
                                  verbose = (args.verbose == 'True'),
                                  alpha = args.alpha, 
                                  beta = args.beta,
                                  gamma = args.gamma,
                                  delta = args.delta, 
                                  epsilon = args.epsilon
                                  zeta = args.zeta)

    env

    for i in range(args.number_of_episodes):
        env.one_episode()
        if args.verbose == 'True':
            print("completed episode number", i + 1)

    env.show_history(save=(args.save_figure == 'True'), moving_average_window=args.figure_convolution_window_size)