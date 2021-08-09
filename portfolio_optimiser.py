import sys
from pathlib import Path
import pandas as pd
from pandas.core.frame import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import scipy.optimize as sco
import scipy.interpolate as sci
import math
np.random.seed(420420)


class PortfolioOptimiser:

    DATALAKE_PATH = Path('datalake')
    ASSET_DATA_FILE_PATH = DATALAKE_PATH / Path('ASX200top10.xlsx')
    CLIENT_DATA_FILE_PATH = DATALAKE_PATH / Path('Client_Details.xlsx')
    OUTPUT_PATH = Path('output')

    risk_free_rate = 0.02

    def __init__(self):
        self.data = {}
        self.client_data = DataFrame()
        self.OUTPUT_PATH.mkdir(exist_ok=True)


    def get_client_risk(self, group: int) -> str:
        if group >= 1 and group <= 3:
            return 'Risk Averse'
        elif group >= 4 and group <= 7:
            return 'Risk Tolerant'
        elif group >= 8 and group <= 10:
            return 'Risk Seeker'
        

    def get_client_age(self, group: int) -> str:
        if group == 1:
            return '18-24'
        elif group == 2:
            return '25-34'
        elif group == 3:
            return '35-44'
        elif group == 4:
            return '45-54'
        elif group == 5:
            return '55-64'


    # extract data from excel sheet
    def ETL(self):
        print('#'*10, 'Extracting Equity Data from Excel','#'*10)

        if self.data:
            self.data.clear()
        if not self.client_data.empty:
            self.client_data = DataFrame()

        # extract list of assets
        asset_list = \
            pd.read_excel(self.ASSET_DATA_FILE_PATH, 'Bloomberg raw', header=None).iloc[0, :].dropna()
        asset_list = asset_list[asset_list != 'AS51 Index']
        asset_list.index = [n for n in range(asset_list.shape[0])]

        # extract data for each asset
        df = pd.read_excel(self.ASSET_DATA_FILE_PATH, 'Bloomberg raw', header=1, index_col=0)

        self.data['returns'] = df.filter(regex='^DAY_TO_DAY_TOT_RETURN_GROSS_DVDS').iloc[:, 1:]
        self.data['returns'].columns = asset_list
        
        self.data['prices'] = df.filter(regex='^PX_LAST').iloc[:, 1:]
        self.data['prices'].columns = asset_list

        self.data['eqy_weighted_prices'] = df.filter(regex='^EQY_WEIGHTED_AVG_PX')
        self.data['eqy_weighted_prices'].columns = asset_list

        self.data['volumes'] = df.filter(regex='^PX_VOLUME')
        self.data['volumes'].columns = asset_list

        self.data['mkt_caps'] = df.filter(regex='^CUR_MKT_CAP')
        self.data['mkt_caps'].columns = asset_list

        print('Extracted: ', self.data.keys())
        print('ASset List:', list(asset_list))

        # extract data for clients
        self.client_data = pd.read_excel(self.CLIENT_DATA_FILE_PATH, 'Data', header=0, index_col=0)
        self.client_data.drop(self.client_data.filter(regex='Unnamed'), axis=1, inplace=True)


    # select relevant features that we will use in our model.
    def feature_engineering(self):
        print()
        print('#'*10, 'Selecting Relevant Features', '#'*10)
        assert self.data, 'No data has been extracted'
        assert not self.client_data.empty, 'No client data has been extracted'

        # select relevant features and remove unwanted features
        features = ['returns']
        print('Selecting features:', features)
        selected_data = {}
        for feature in features:
            assert self.is_normal(feature), 'feature not drawn from a normal distribution'
            selected_data[feature] = self.data[feature]
        self.data = selected_data

        print('-'*20)

        # select random client and remove all other clients
        client_id = np.random.randint(self.client_data.index[0], self.client_data.index[-1]+1)
        self.client_data = self.client_data.loc[client_id, :]
        self.client_data.name = client_id

        risk_group = self.client_data.iloc[0]
        age_group = self.client_data.iloc[1]
        weights = self.client_data.iloc[2:]

        self.save_portfolio(weights, 'portfolio_client')

        print('Randomly selected 1 client\'s portfolios to optimise...')
        print('Client:      ', self.client_data.name)
        print('Risk Profile:', risk_group, '('+self.get_client_risk(risk_group)+')')
        print('Age Group:   ', self.get_client_age(age_group))
        print('-'*20)
        print('Client Holdings:')
        print(weights.to_string())
        print('-'*20)

        client_return = self.portfolio_return(weights, self.data['returns'])
        client_stdev = self.portfolio_stdev(weights, self.data['returns'])
        client_sharpe = self.portfolio_sharpe(weights, self.data['returns'])
        print('Current return:    ', str(round(100*client_return, 3))+'%')
        print('Current volatility:', round(client_stdev, 3))
        print('Current Sharpe:    ', round(client_sharpe, 4))


    # test normality of feature
    def is_normal(self, feature: str) -> bool:
        print('Testing Normality of', feature)

        # graphical inspection using density plot
        self.data[feature].plot.kde(subplots=True, figsize = [10, 20])
        plt.savefig(self.OUTPUT_PATH / Path(feature+'_distribution.png'))
        plt.close()

        for stock, stock_data in self.data[feature].items():
            is_normal = False

            # statistical test for normality
            if not is_normal and scs.normaltest(stock_data).pvalue > 0.05:
                is_normal = True
            
            # sample size test for normality
            # central limit theorem states that a distribution will approach normality as size of sample increases
            if not is_normal and stock_data.shape[0] > 30:
                is_normal = True

            if is_normal:
                print('normal distribution passed', '('+stock, feature+')')
            else:
                print('normal distribution failed', '('+stock, feature+')')
                return False
        
        return True

    # run and test model using simulated returns data
    def model_design(self):
        print()
        print('#'*10, 'Testing Portfolio Optimiser using Simulated Return Data', '#'*10)
        assert 'returns' in self.data, 'Returns have not been selected as a relevant feature'

        prices = {
            'Risky' : 50*np.exp((-0.01-0.5*0.78**2)*1+0.78*np.sqrt(1)*np.random.standard_normal(365)),
            'Flat' : 10*np.exp((-0.01-0.5*0.01**2)*1+0.01*np.sqrt(1)*np.random.standard_normal(365)),
            'SmallCap' : 10*np.exp((-0.01-0.5*0.5**2)*1+0.5*np.sqrt(1)*np.random.standard_normal(365)),
            'LargeCap' : 100*np.exp((-0.01-0.5*0.1**2)*1+0.1*np.sqrt(1)*np.random.standard_normal(365)),
            'Gold' : 1000*np.exp((-0.01-0.5*0.05**2)*1+0.05*np.sqrt(1)*np.random.standard_normal(365))
        }
        prices = pd.DataFrame(prices)
        prices.index = pd.date_range('2019-1-1', periods=365, freq='D')
        returns = np.log(prices / prices.shift(1)).iloc[1:,:]*100

        self.portfolio_optimiser(
            returns = returns, 
            client_weights = np.random.dirichlet([1]*len(returns.columns)),
            output_suffix = '_simulated'
        )

    # run model using our actual data
    def model_implementation(self):
        print()
        print('#'*10, 'Running Portfolio Optimiser using Extracted Return Data', '#'*10)
        assert 'returns' in self.data, 'Returns have not been selected as a relevant feature'

        self.portfolio_optimiser(
            returns = self.data['returns'],
            client_weights = self.client_data.iloc[2:]
        )

    # uses modern portfolio theory to assign portfolio weights
    # num_simulations controls the number of portfolio simulations to create
    def portfolio_optimiser(self, returns: DataFrame, client_weights: list, output_suffix: str = ''):
        n_assets = returns.shape[1]        

        '''
        SIMULATING 2500 RANDOM PORTFOLIO WEIGHT COMBINATIONS
        '''

        simulated_returns = []
        simulated_stdevs = []
        for _ in range(2500):
            weights = np.random.dirichlet([0.3]*n_assets)
            simulated_returns.append(self.portfolio_return(weights, returns))
            simulated_stdevs.append(self.portfolio_stdev(weights, returns))

        simulated_returns = np.array(simulated_returns)
        simulated_stdevs = np.array(simulated_stdevs)

        '''
        CALCULATING OPTIMAL PORTFOLIO'S
        '''

        # portfolio allocation when minimising volatility
        portfolio_alloc_stdev = sco.minimize(
            fun = self.portfolio_stdev,
            x0 = [1/n_assets for _ in range(n_assets)], 
            args = returns, 
            method = 'SLSQP',
            bounds = [(0, 1) for _ in range(n_assets)], 
            constraints = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        )
        assert portfolio_alloc_stdev['success'], portfolio_alloc_stdev['message']

        optimal_weights_volatility = portfolio_alloc_stdev['x']
        optimal_return_volatility = self.portfolio_return(optimal_weights_volatility, returns)
        optimal_stdev_volatility = self.portfolio_stdev(optimal_weights_volatility, returns)
        optimal_sharpe_volatility = \
            self.portfolio_sharpe(optimal_weights_volatility, returns)

        print('Portfolio Allocation - Minimising Volatility:')
        for idx, weight in enumerate(optimal_weights_volatility):
            print(returns.columns[idx]+':', str(round(100*weight, 3))+'%')
        print('-'*20)
        print('Estimated Return:    ', str(round(100*optimal_return_volatility, 3))+'%')
        print('Estimated Volatility:', round(optimal_stdev_volatility, 3))
        print('Sharpe Ratio:        ', round(optimal_sharpe_volatility, 4))
        print('-'*20)

        # portfolio allocation when maximising sharpe ratio
        portfolio_alloc_sharpe = sco.minimize(
            fun = lambda weights: -self.portfolio_sharpe(weights, returns),
            x0 = [1/n_assets for _ in range(n_assets)], 
            method = 'SLSQP',
            bounds = [(0, 1) for _ in range(n_assets)], 
            constraints = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        )
        assert portfolio_alloc_sharpe['success'], portfolio_alloc_sharpe['message']

        optimal_weights_sharpe = portfolio_alloc_sharpe['x']
        optimal_return_sharpe = self.portfolio_return(optimal_weights_sharpe, returns)
        optimal_stdev_sharpe = self.portfolio_stdev(optimal_weights_sharpe, returns)
        optimal_sharpe_sharpe = \
            self.portfolio_sharpe(optimal_weights_sharpe, returns)

        print('Portfolio Allocation - Maximising Sharpe:')
        for idx, weight in enumerate(optimal_weights_sharpe):
            print(returns.columns[idx]+':', str(round(100*weight, 3))+'%')
        print('-'*20)
        print('Estimated Return:    ', str(round(100*optimal_return_sharpe, 3))+'%')
        print('Estimated Volatility:', round(optimal_stdev_sharpe, 3))
        print('Sharpe Ratio:        ', round(optimal_sharpe_sharpe, 4))

        # find efficient frontier for returns in range [0.001, 0.14]
        efficient_returns = np.linspace(simulated_returns.min(), simulated_returns.max(), 50)
        efficient_stdevs = np.array([
            sco.minimize(
                fun = self.portfolio_stdev, 
                x0 = [1/n_assets for _ in range(n_assets)], 
                args = returns,
                method = 'SLSQP', 
                bounds = tuple((0, 1) for _ in range(n_assets)), 
                constraints = (
                    {'type': 'eq', 'fun': lambda x:  self.portfolio_return(x, returns) - efficient_return},
                    {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}
                )
            )['fun'] for efficient_return in efficient_returns
        ])
        min_stdev = efficient_stdevs.argmin()
        efficient_returns = efficient_returns[min_stdev:]
        efficient_stdevs = efficient_stdevs[min_stdev:]

        # generate spline curve of efficient frontier
        spline = sci.splrep(efficient_stdevs, efficient_returns)
        efficient_returns = sci.splev(efficient_stdevs, spline)

        '''
        Calculate client portfolio statistics
        '''

        client_return = self.portfolio_return(client_weights, returns)
        client_stdev = self.portfolio_stdev(client_weights, returns)

        '''
        PLOTTING MODEL RESULTS AND SAVE TABLES
        '''
        if not output_suffix:
            self.save_portfolio(client_weights, 'portfolio_client')
            self.save_portfolio(optimal_weights_sharpe, 'portfolio_optimal_sharpe')
            self.save_portfolio(optimal_weights_volatility, 'portfolio_minimum_volatility')

        plt.figure(figsize=(16, 8))
        # plot return and volatility of simulated portfolios
        plt.scatter(
            x = simulated_stdevs, 
            y = simulated_returns*100, 
            c = (simulated_returns - self.risk_free_rate) / simulated_stdevs,
            marker = 'o', 
        )
        # plot efficient frontier
        plt.plot(
            efficient_stdevs, 
            efficient_returns*100,
            label = 'Efficient Frontier'
        )
        # plot portfolio with lowest volatility
        plt.plot(
            optimal_stdev_volatility, 
            optimal_return_volatility*100,
            'k*',
            markersize = 15.0,
            label = 'Lowest Volatility'
        )
        # plot portfolio with highest sharpe ratio
        plt.plot(
            optimal_stdev_sharpe, 
            optimal_return_sharpe*100,
            'b*',
            markersize = 15.0, 
            label = 'Optimal Sharpe Ratio'
        )
        # plot client portfolio
        plt.plot(
            client_stdev,
            client_return*100,
            'ro',
            markersize = 10.0, 
            label = 'Current Portfolio'
        )
        # plot information
        plt.title('Client '+str(self.client_data.name)+' Portfolio Simulation', {'size': 20})
        plt.legend(loc = 'upper right')
        plt.grid(True)
        plt.xlabel('Expected volatility')
        plt.ylabel('Expected return (%)')
        plt.colorbar(label = 'Sharpe ratio')
        path = self.OUTPUT_PATH / Path('optimal_portfolios'+output_suffix+'.png')
        plt.savefig(path)
        plt.close()


    def portfolio_return(self, weights, returns) -> float:
        try:
            return np.sum(returns.mean() * weights)
        except ValueError:
            raise ValueError('Number of weights and assets must match')
    

    def portfolio_stdev(self, weights, returns) -> float:
        try:
            return np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
        except ValueError:
            raise ValueError('Number of weights and assets must match')


    def portfolio_sharpe(self, weights, returns) -> float:
        try:
            return (self.portfolio_return(weights, returns) - self.risk_free_rate) / self.portfolio_stdev(weights, returns)
        except ValueError:
            raise ValueError('Number of weights and assets must match')


    def save_portfolio(self, weights, name):
        with pd.ExcelWriter(self.OUTPUT_PATH / Path(name+'.xlsx')) as writer:
            Series(weights, index=self.data['returns'].columns, name='Weights (%)') \
                .map(lambda x: round(x*100, 2)) \
                .to_excel(writer)