import numpy as np
from agent import Agent

class Environment:
    """Environment of agents in different countries making transactions. 
    Each transaction is done in only one currency.
        
    Arguments:
    ----------
    
    population_size: int
    number_of_countries: int
    number_of_currencies: int
    number_of_transactions: int
    increase_weakest_currency: bool
        whether to increase the weakest currency over all the episodes
    even_countries_currency_spread: bool
        whether there should be an evenly distributed spread of assigned currencies between countries
    verbose: bool
        whether to log progress of the simulation
    alpha: float
        how much more money is an agent expected to have in his home currency
    beta: float
        what maximum percent of his budget is an agent willing to use for a transaction 
    gamma: float
        how much more likely is an agent to make a transaction  with someone from their own country
    delta: float
        how much more impact on the probability of choosing currencies does their value have 
    epsilon: float
        how much impact on the probability of choosing currencies does agent's wallet contents have 
    zeta: float
        how much impact on the probability of choosing currencies does the fact that the seller is from the same country has
    """

    def __init__(self, population_size: int = 1000,
                       number_of_countries: int = 4,
                       number_of_currencies: int = 4,
                       number_of_transactions: int = 100,
                       number_of_episodes: int = 1000,
                       even_countries_currency_spread: bool = True,
                       increase_weakest_currency: bool = True,
                       verbose: bool = True,
                       alpha: float = 2,
                       beta: float = 0.5,
                       gamma: float = 0.0086,
                       delta: float = 1.5,
                       epsilon: float = 0.5,
                       zeta: float = 4):
        
        if number_of_currencies > number_of_countries:
            raise ValueError("number_of_currencies > number_of_countries")

        self.population_size = population_size
        self.number_of_countries = number_of_countries
        self.number_of_currencies = number_of_currencies
        self.number_of_transactions = number_of_transactions
        self.increase_weakest_currency = increase_weakest_currency
        self.verbose = verbose
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.even_countries_currency_spread = even_countries_currency_spread

        self.currency_exchange_matrix = None
        self.create_currency_exchange()

        self.increment_currency = np.argmax([self.currency_exchange_matrix[c][0] for c in range(self.number_of_currencies)])
        weakest_currency_value = self.currency_exchange_matrix[self.increment_currency][0]
        strongest_currency_value = np.min([self.currency_exchange_matrix[c][0] for c in range(self.number_of_currencies)])
        
        self.increment_currency_value = 2 * (strongest_currency_value - weakest_currency_value) / number_of_episodes

        # the lower the value, the higher the probability of making a transaction in this currency
        sum_of_inverses = sum( 1 / (value ** self.delta) for value in self.currency_exchange_matrix[0])
        self.probabilities_of_choosing_currencies = [1 / ((value ** self.delta)  * sum_of_inverses) for value in self.currency_exchange_matrix[0]]

        self.countries_currencies = None 
        if self.even_countries_currency_spread:
            self.countries_currencies = [i % self.number_of_currencies for i in range(self.number_of_countries)]
        else:
            if self.number_of_countries == self.number_of_currencies:
                self.countries_currencies = np.random.permutation(self.number_of_currencies)
            else:
                diff = self.number_of_countries - self.number_of_currencies
                self.countries_currencies = np.concatenate((np.random.permutation(self.number_of_currencies), 
                                                            np.random.randint(self.number_of_currencies, size=diff)) )
        self.countries_currencies = sorted(self.countries_currencies)

        self.probabilities_of_choosing_countries = [1 / (self.number_of_countries + (self.gamma - 1)) for _ in range(self.number_of_countries)]
        self.home_country_probability = self.gamma * self.probabilities_of_choosing_countries[0]

        self.agents = [Agent(self.number_of_currencies, self.number_of_countries, self.countries_currencies, self.alpha) for _ in range(population_size)]
        self.history_of_agents = [self.agents]

        self.history_of_total_value_of_transactions = []

        self.transactions_total_record = np.zeros((self.number_of_countries, self.number_of_countries, self.number_of_currencies))

    def create_currency_exchange(self):
        """
            Usage: 
            >>> self.currency_exchange_matrix[FROM][TO]
        """

        # TODO: implement constructive method of creating an exchange for any number of currencies
        # (randomly pick exchange rates for: C_1 - C_2, C_2 - C_3, ... C_{n-1} - C_{n})
        # (find C_i - C_j by finding a shortest, already established path)

        if self.number_of_currencies == 4:
            
            # exchange rates taken as of 3/10/2023
            # USD, EUR, GBP, CHF
            self.currency_exchange_matrix = [ [1,    0.95, 0.83, 0.92], 
                                              [1.05, 1,    0.87, 0.96], 
                                              [1.21, 1.15, 1,    1.11],
                                              [1.09, 1.04, 0.9,  1   ] ]

        else:
            raise NotImplementedError("Change the number of currencies")

        # ensure its balanced
        for i in range(self.number_of_currencies):
            for j in range(i, self.number_of_currencies):
                self.currency_exchange_matrix[i][j] = 1 / self.currency_exchange_matrix[j][i]


    def one_episode(self):
        """
            series of random transactions
        """

        episode_total_value_of_transactions = [0] * self.number_of_currencies

        exchange_rate_to_primary_currency = [self.currency_exchange_matrix[i][0] for i in range(self.number_of_currencies)]

        for _ in range(self.number_of_transactions):
            
            buyer_index, buyer, seller_index, seller = self.choose_agents_for_transaction()

            chosen_currency, transaction_value_in_chosen_currency = buyer.choose_currency_and_transaction_value(exchange_rate_to_primary_currency, self.beta, self.delta, self.epsilon, self.zeta, (buyer.country_id == seller.country_id))

            # record it for future plotting
            episode_total_value_of_transactions[chosen_currency] += transaction_value_in_chosen_currency

            # actual transfer of money
            self.agents[buyer_index].wallet[chosen_currency] -= transaction_value_in_chosen_currency
            self.agents[seller_index].wallet[chosen_currency] += transaction_value_in_chosen_currency

            self.transactions_total_record[buyer.country_id][seller.country_id][chosen_currency] += transaction_value_in_chosen_currency

        self.history_of_total_value_of_transactions.append(episode_total_value_of_transactions)
        self.history_of_agents.append(self.agents)

        if self.increase_weakest_currency:
            self.update_currency_exchange()

    def choose_agents_for_transaction(self):
        
        # randomly choose an agent
        buyer_index = np.random.randint(self.population_size)
        buyer = self.agents[buyer_index]

        # choose a country
        probabilities = self.probabilities_of_choosing_countries.copy()
        probabilities[buyer.country_id] = self.home_country_probability
        chosen_country = np.random.choice(self.number_of_countries, p = probabilities) 

        relevant_agent_indexes = []
        for i, agent in enumerate(self.agents):
            if agent.country_id == chosen_country:
                relevant_agent_indexes.append(i)

        seller_index = np.random.choice(relevant_agent_indexes)
        seller = self.agents[seller_index]

        return buyer_index, buyer, seller_index, seller

    def update_currency_exchange(self):
        
        before = self.currency_exchange_matrix[self.increment_currency][0]
        self.currency_exchange_matrix[self.increment_currency][0] += self.increment_currency_value
        relative_change = self.currency_exchange_matrix[self.increment_currency][0] / before

        for i in range(1, self.number_of_currencies):

            if i == self.increment_currency:
                continue
            
            self.currency_exchange_matrix[self.increment_currency][i] *= relative_change

        for i in range(self.number_of_currencies):
            self.currency_exchange_matrix[i][self.increment_currency] = 1 / self.currency_exchange_matrix[self.increment_currency][i]

    def show_history(self, save: bool = False, moving_average_window: int = 10):
        """
            Total value of transactions in each currency plotted through time.
            Additional line showing how balance (respective to their relative value) changes.

            Saving a figure saves it with today's datetime and hour.

            save: bool
            moving_average_window: int
        """

        lines = [[] for _ in range(self.number_of_currencies)]

        how_many_agents_have_this_currency = [0] * self.number_of_currencies
        for agent in self.agents:
            currency = self.countries_currencies[agent.country_id]
            how_many_agents_have_this_currency[currency] += 1

        standard_deviation_line = []
        for episode in self.history_of_total_value_of_transactions:
            for i, value in enumerate(episode):
                lines[i].append(value)

            var = 0
            tot_inverse_currency_values = sum(self.currency_exchange_matrix[0])
            tot_episode_currency_values = sum(1 / value for value in episode)
            for i, value in enumerate(episode):
                var += ((1 / self.currency_exchange_matrix[0][i]) / tot_inverse_currency_values - value / tot_episode_currency_values) ** 2
            standard_deviation_from_balanced_transaction_values = np.sqrt(var)

            standard_deviation_line.append(standard_deviation_from_balanced_transaction_values)

        for i, line in enumerate(lines):
            lines[i] = np.convolve(line, np.ones(moving_average_window)/moving_average_window, mode='valid') / how_many_agents_have_this_currency[i]
        standard_deviation_line = np.convolve(standard_deviation_line, np.ones(moving_average_window)/moving_average_window, mode='valid')

        import matplotlib.pyplot as plt

        # Create a figure with subplots
        # fig, axes = plt.subplots(2, self.number_of_currencies, figsize=(15, 7), layout='constrained')
        fig = plt.figure(figsize=(15, 7), layout='constrained')

        spec = fig.add_gridspec(ncols=self.number_of_currencies, nrows=2)
        fig.set_size_inches(15, 10)
        fig.set_constrained_layout_pads(w_pad=0.8, h_pad=0.25, wspace=0, hspace=0)

        ax0 = fig.add_subplot(spec[0, :])
        other_axes = [fig.add_subplot(spec[1, k]) for k in range(self.number_of_currencies)]
        ax1 = ax0.twinx()
        ax1.set_ylabel("Odchylenie standardowe")

        matrices = []
        for i in range(self.number_of_currencies):
            matrix = np.zeros((self.number_of_countries, self.number_of_countries))

            for x in range(self.number_of_countries):
                for y in range(self.number_of_countries):
                    s = sum(self.transactions_total_record[x][y])
                    matrix[x][y] = 0 if s == 0 else self.transactions_total_record[x][y][i] / s

            matrices.append(matrix)

        vmin = np.amin(matrices)
        vmax = np.amax(matrices)
        # Loop through the number of matrices
        for i, matrix in enumerate(matrices):
            # Create the matrix diagram in the bottom row
            other_axes[i].imshow(matrix, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
            other_axes[i].set_title(f'Waluta {i}')

        ax0.set_xlabel("Numer epizodu")
        ax0.set_ylabel("wartość transakcji")

        ax1.plot(standard_deviation_line, label = "Odchylenie standardowe od idealnego\n rozkładu cen", linestyle='--', color='black')

        ax0.set_title("Całkowita wartość transakcji w danej walucie w każdym epizodzie\npodzielona przez ilość agentów dla których jest to waluta domowa")
        for i in range(self.number_of_currencies):
            ax0.plot(lines[i], label=f"Waluta {i} (wartość {self.currency_exchange_matrix[0][i]:.3f}, średnia {np.mean(lines[i]):.3f})", alpha=0.5)

        ax1.set_ylim(top=2*max(standard_deviation_line))

        ax0.legend(loc='upper right', framealpha=1)
        ax1.legend(loc='upper left', framealpha=1)

        parameter_text = f"""
                            Parametry:
                                waluta rośnie = {'Tak' if self.increase_weakest_currency else 'Nie'}
                                rozmiar konwolucji = {moving_average_window}
                                populacja = {self.population_size}
                                kraje = {self.number_of_countries}
                                waluty = {self.number_of_currencies}
                                transakcje = {self.number_of_transactions}
                                alpha = {self.alpha}
                                beta = {self.beta}
                                gamma = {self.gamma}
                                delta = {self.delta}
                                epsilon = {self.epsilon}
                                zeta = {self.zeta}

                            Kraje:
                           """
        
        for i, currency in enumerate(self.countries_currencies):
            parameter_text += f'kraj {i} - waluta {currency}\n'

        fig.text(.5, .05, 'Kolor w pozycji (x, y) w każdej macierzy relacji dla każdej waluty reprezentuje \n jaką proporcję wszystkich transakcji z państwa y do państwa x są transakcje w tej walucie.', ha='center')
        plt.subplots_adjust(right = 0.8, wspace=0.3, hspace=0.1)
        plt.figtext(0.97, 0.5, parameter_text, va='center', ha='right', fontsize=10)

        if save:    
            from datetime import datetime
            plt.savefig('experiment images\eksperyment z macierzą relacji i zeta ' + str(datetime.now()).replace(':', '').replace('.', '') + '.png')
        else:
            plt.show()  