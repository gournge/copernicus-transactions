# copernicus-transactions
Simulation aiming to research whether the Copernicus-Gresham's law defined by Nikolas Copernicus (in some shape or form) holds in a multi-currency market.

## Running the simulation 

```
python \.simulate.py --population_size 1000
                     --number_of_countries 4
                     --number_of_currencies 4
                     --number_of_transactions 100
                     --even_countries_currencies_spread True
                     --verbose True
                     --alpha 2
                     --beta 0.5
                     --gamma 2
                     --delta 1.5
                     --epsilon 0.5
                     --number_of_episodes 500
                     --save_figure False
                     --figure_convolution_window_size 10
```

or type 

```
python \.simulate.py --help
```

for additional information.

## Notes

`delta = 1 / 47.15% = 2.12 ` makes sense as 47.15% is the average percent of export in relation to GDP (source: https://www.theglobaleconomy.com/rankings/exports/)