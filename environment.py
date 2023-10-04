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
    verbose: bool
        whether to log progress of the simulation
    alpha: float
        how much more money is an agent expected to have 
        in his home currency
    beta: float
        what maximum percent of his budget 
        is an agent willing to use for a transaction 
    gamma: float
        how much more likely is an agent to make a transaction 
        with someone from their own country
    delta: float
        how much more impact on the probability of choosing currencies does their value have 
    epsilon: float
        how much impact on the probability of choosing currencies does agent's wallet contents have 
        
    """

    def __init__(self, population_size: int = 1000,
                       number_of_countries: int = 4,
                       number_of_currencies: int = 4,
                       number_of_transactions: int = 100,
                       verbose: bool = True,
                       alpha: float = 2,
                       beta: float = 0.5,
                       gamma: float = 2,
                       delta: float = 1.5,
                       epsilon: float = 0.5):

        
        if number_of_currencies > number_of_countries:
            raise ValueError("number_of_currencies > number_of_countries")

        self.population_size = population_size
        self.number_of_countries = number_of_countries
        self.number_of_currencies = number_of_currencies
        self.number_of_transactions = number_of_transactions
        self.verbose = verbose
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

        self.currency_exchange_matrix = None
        self.create_currency_exchange()

        # the lower the value, the higher the probability of making a transaction in this currency
        sum_of_inverses = sum( 1 / (value ** self.delta) for value in self.currency_exchange_matrix[0])
        self.probabilities_of_choosing_currencies = [1 / ((value ** self.delta)  * sum_of_inverses) for value in self.currency_exchange_matrix[0]]

        self.countries_currencies = None 
        if self.number_of_countries == self.number_of_currencies:
            self.countries_currencies = np.random.permutation(self.number_of_currencies)
        else:
            diff = self.number_of_countries - self.number_of_currencies
            self.countries_currencies = np.concatenate((np.random.permutation(self.number_of_currencies), 
                                                        np.random.randint(self.number_of_currencies, size=diff)) )

        self.probabilities_of_choosing_countries = [1 / (self.number_of_countries + (self.gamma - 1)) for _ in range(self.number_of_countries)]
        self.home_country_probability = self.gamma * self.probabilities_of_choosing_countries[0]

        self.agents = [Agent(self.number_of_currencies, self.number_of_countries, self.countries_currencies, self.alpha) for _ in range(population_size)]
        self.history_of_agents = [self.agents]

        self.history_of_total_value_of_transactions = []

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
                                              [1.21, 0.15, 1,    1.11],
                                              [1.09, 1.04, 0.9,  1   ] ]

        else:
            raise NotImplementedError("Change the number of currencies")


    def one_episode(self):
        """
            series of random transactions
        """

        episode_total_value_of_transactions = [0] * self.number_of_currencies

        for _ in range(self.number_of_transactions):
            
            buyer_index, buyer, seller_index, seller = self.choose_agents_for_transaction()

            # valued by the first currency for simplicity
            buyer_total_budget = seller.wallet[0]
            for currency in range(1, self.number_of_currencies):
                buyer_total_budget += self.currency_exchange_matrix[0][currency] * buyer.wallet[currency]

            chosen_currency = None
            if self.epsilon == 0:
                chosen_currency = np.random.choice(self.number_of_currencies, p=self.probabilities_of_choosing_currencies)
            else:
                chosen_currency = buyer.choose_currency(self.currency_exchange_matrix[0], self.delta, self.epsilon)

            maximum_transaction_value = min( self.currency_exchange_matrix[0][chosen_currency] * (buyer_total_budget * self.beta), buyer.wallet[chosen_currency])

            transaction_value_in_chosen_currency = np.random.rand() * maximum_transaction_value

            # record it for future plotting
            transaction_value = self.currency_exchange_matrix[chosen_currency][0] * transaction_value_in_chosen_currency
            episode_total_value_of_transactions[chosen_currency] += transaction_value_in_chosen_currency

            # actual transfer of money
            self.agents[buyer_index].wallet[chosen_currency] -= transaction_value_in_chosen_currency
            self.agents[seller_index].wallet[chosen_currency] += transaction_value_in_chosen_currency

        self.history_of_total_value_of_transactions.append(episode_total_value_of_transactions)
        self.history_of_agents.append(self.agents)

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

    def show_history(self, save: bool = False, moving_average_window = 10):
        """
            Total value of transactions in each currency plotted through time.
            Additional line showing how balance (respective to their relative value) changes.

            Saving a figure saves it with today's datetime and hour.
        """

        lines = [[] for _ in range(self.number_of_currencies)]

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
            lines[i] = np.convolve(line, np.ones(moving_average_window)/moving_average_window, mode='valid')
        standard_deviation_line = np.convolve(standard_deviation_line, np.ones(moving_average_window)/moving_average_window, mode='valid')

        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()
        fig.set_size_inches(15, 10)

        ax1.set_xlabel("Episode number")
        ax1.set_ylabel("Transaction value")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Standard deviation")

        ax2.plot(standard_deviation_line, label = "Standard deviation from a perfect\ntransaction value distribution", linestyle='--', color='black')

        plt.title("Total value of transactions in each episode")
        for i in range(self.number_of_currencies):
            ax1.plot(lines[i], label=f"Currency {i} (value {self.currency_exchange_matrix[0][i]})")
        ax1.legend(loc='upper right', framealpha=1)
        ax2.legend(loc='upper left', framealpha=1)

        parameter_text = f"""
                            Parameters:
                                convolution size = {moving_average_window}
                                population = {self.population_size}
                                countries = {self.number_of_countries}
                                currencies = {self.number_of_currencies}
                                transactions = {self.number_of_transactions}
                                alpha = {self.alpha}
                                beta = {self.beta}
                                gamma = {self.gamma}
                                delta = {self.delta}
                                epsilon = {self.epsilon}
                           """

        plt.tight_layout()
        plt.subplots_adjust(right = 0.8)
        plt.figtext(0.97, 0.5, parameter_text, va='center', ha='right', fontsize=10)

        if save:
            from datetime import datetime
            plt.savefig('experiment images\experiment ' + str(datetime.now()).replace(':', '').replace('.', '') + '.png')
        else:
            plt.show()  

