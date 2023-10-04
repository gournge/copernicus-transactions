import numpy as np

class Agent:
    """Agent with a wallet that is more initialised with values biased towards his home currency
    (has higher value in that currency)

    number_of_currencies: int
    number_of_countries: int
    beta: float 
        how much more money is an agent expected to have 
        in his home currency

    """

    def __init__(self, number_of_currencies: int, 
                       number_of_countries: int, 
                       countries_currencies: np.ndarray,
                       alpha: float):
        
        self.country_id = np.random.randint(number_of_countries)
        self.wallet = np.array( [np.random.rand() for _ in range(number_of_currencies)] )

        home_currency = countries_currencies[self.country_id]        

        self.wallet[home_currency] *= alpha

    def choose_currency(self, currency_values: list[float], delta: float, epsilon: float):
        """
        
        currency_values: list of floats
        delta: float
            how much more impact on the probability of choosing currencies does their value have 
        epsilon: float
            how much impact on the probability of choosing currencies does agent's wallet contents have 

        """

        perceived_currency_values = []
        for i, currency_value in enumerate(currency_values):

            wallet_currency_value = currency_value * self.wallet[i]

            perceived_currency_values.append( (wallet_currency_value ** epsilon) * (currency_value ** delta) )
                
        # the lower the value the higher the probability 
        s = sum( 1 / value for value in perceived_currency_values)
        probabilities = [1 / (value * s) for value in perceived_currency_values]

        return np.random.choice(len(currency_values), p = probabilities)