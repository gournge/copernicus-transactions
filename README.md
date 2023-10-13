# copernicus-transactions
Simulation aiming to research whether the Copernicus-Gresham's law defined by Nikolas Copernicus (in some shape or form) holds in a multi-currency market. The model is described in depth on [my website](https://gournge.github.io/posts/Simulating-the-Copernicus-Gresham's-law-in-currency-markets/).

## Running the simulation 

```
python \.simulate.py --population_size 1000
                     --number_of_countries 4
                     --number_of_currencies 4
                     --even_countries_currencies_spread True
                     --number_of_transactions 100
                     --number_of_episodes 500
                     --figure_convolution_window_size 10
                     --save_figure False
                     --verbose True
                     --increase_weakest_currency True
                     --alpha 2
                     --beta 0.5
                     --gamma 0.0086
                     --delta 1.5
                     --epsilon 0.5
                     --zeta 10
```

or type 

```
python \.simulate.py --help
```

for additional information.

## Notes

`gamma = 100% - 47.15% - 51.99% / 100% = 0.0086 ` makes sense as 47.15% is the average percent of export in relation to GDP and 51.99% is the average import rate (source: https://www.theglobaleconomy.com/rankings/exports/)