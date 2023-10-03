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
                       beta: float):
        
        self.country_id = np.random.randint(number_of_countries)
        self.wallet = np.array( [np.random.rand() for _ in range(number_of_currencies)] )

        home_currency = countries_currencies[self.country_id]        

        self.wallet[home_currency] *= beta
