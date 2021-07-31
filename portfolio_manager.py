import sys
from pathlib import Path
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import scipy.optimize as sco
import scipy.interpolate as sci
import math
np.random.seed(420)


class PortfolioManager:

    DATALAKE_PATH = Path('datalake')
    ASSET_DATA_FILE_PATH = DATALAKE_PATH / Path('ASX200top10.xlsx')
    CLIENT_DATA_FILE_PATH = DATALAKE_PATH / Path('Client_Details.xlsx')
    OUTPUT_PATH = Path('output')

    risk_free_rate = 0.02


    def __init__(self):
        self.data = {}
        self.asset_list = None
        self.all_clients = None
        self.client_id = None
        self.client_data = None
        self.selected_features = []
        self.OUTPUT_PATH.mkdir(exist_ok=True)


    def get_client_risk(self, group: int):
        if group >= 1 and group <= 3:
            return 'Risk Averse'
        elif group >= 4 and group <= 7:
            return 'Risk Tolerant'
        elif group >= 8 and group <= 10:
            return 'Risk Seeker'
        

    def get_client_age(self, group: int):
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
        print('#'*10, 'Extracting data from excel','#'*10)

        if self.data:
            self.data.clear()
        if self.asset_list:
            self.asset_list = None
        if self.all_clients:
            self.all_clients = None

        # extract list of assets
        self.asset_list = \
            pd.read_excel(self.ASSET_DATA_FILE_PATH, 'Bloomberg raw', header=None).iloc[0, :].dropna()
        self.asset_list = self.asset_list[self.asset_list != 'AS51 Index']
        self.asset_list.index = [n for n in range(self.asset_list.shape[0])]
        print('Asset list:', list(self.asset_list))

        # extract data for each asset
        df = pd.read_excel(self.ASSET_DATA_FILE_PATH, 'Bloomberg raw', header=1, index_col=0)
        sys.stdout.write('Data extracted:')

        self.data['returns'] = df.filter(regex='^DAY_TO_DAY_TOT_RETURN_GROSS_DVDS').iloc[:, 1:]
        self.data['returns'].columns = self.asset_list
        sys.stdout.write(' returns')
        
        self.data['prices'] = df.filter(regex='^PX_LAST').iloc[:, 1:]
        self.data['prices'].columns = self.asset_list
        sys.stdout.write(', prices')

        self.data['eqy_weighted_prices'] = df.filter(regex='^EQY_WEIGHTED_AVG_PX')
        self.data['eqy_weighted_prices'].columns = self.asset_list
        sys.stdout.write(', equity weighted prices')

        self.data['volumes'] = df.filter(regex='^PX_VOLUME')
        self.data['volumes'].columns = self.asset_list
        sys.stdout.write(', volumes')

        self.data['mkt_caps'] = df.filter(regex='^CUR_MKT_CAP')
        self.data['mkt_caps'].columns = self.asset_list
        sys.stdout.write(', market caps')

        # extract data for clients
        self.all_clients = pd.read_excel(self.CLIENT_DATA_FILE_PATH, 'Data', header=0, index_col=0)
        self.all_clients.drop(self.all_clients.filter(regex='Unnamed'), axis=1, inplace=True)
        sys.stdout.write(', client data')
        print()


    # select relevant features that we will use in our model.
    def feature_engineering(self):
        print()
        print('#'*10, 'Selecting relevant features', '#'*10)
        assert self.data, 'No data has been extracted'

        if self.selected_features:
            self.selected_features.clear()
        if self.client_data:
            self.client_data.clear()
        
        features = ['returns']
        for feature in features:
            if self.is_normal(feature):
                self.selected_features.append(feature)

        print('Selected features:', self.selected_features)
        print('-'*30)

        print('Randomly selecting 1 client\'s portfolios to optimise...')
        self.client_id = np.random.randint(self.all_clients.index[0], self.all_clients.index[-1]+1)
        self.client_data = self.all_clients.loc[self.client_id, :]

        risk_group = self.client_data.iloc[0]
        age_group = self.client_data.iloc[1]
        weights = self.client_data.iloc[2:]

        print('Client:'+' '*6, self.client_id)
        print('Risk Profile:', risk_group, '('+self.get_client_risk(risk_group)+')')
        print('Age Group:'+' '*3, self.get_client_age(age_group))
        print('-'*20)
        print('Client Holdings:')
        print(weights.to_string())
        print('-'*20)

        client_return = self.portfolio_return(weights, self.data['returns'])
        client_stdev = self.portfolio_stdev(weights, self.data['returns'])
        print('Current return:', str(round(100*client_return, 3))+'%')
        print('Current volatility:', round(client_stdev, 3))


    # test normality of feature
    def is_normal(self, feature: str, output_graphs: bool = True):
        print('Testing Normality of', feature)

        # graphical inspection using density plot
        self.data[feature].plot.kde(subplots=True, figsize = [10, 15])
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


    # uses modern portfolio theory to assign portfolio weights
    # num_simulations controls the number of portfolio simulations to create
    def model_design(self):
        print()
        print('#'*10, 'Calculating portfolio weights', '#'*10)
        assert self.selected_features and 'returns' in self.selected_features, \
            'Returns have not been selected as a relevant feature'

        returns = self.data['returns']
        n_assets = returns.shape[1]

        '''
        SIMULATING 2500 RANDOM PORTFOLIO WEIGHT COMBINATIONS
        '''

        simulated_returns = []
        simulated_stdevs = []
        for _ in range(2500):
            weights = np.random.dirichlet([0.25]*n_assets)
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

        print('Portfolio Allocation (Minimising Volatility):')
        for idx, weight in enumerate(optimal_weights_volatility):
            print(self.asset_list[idx]+':', str(round(100*weight, 3))+'%')
        print('-'*20)
        print('Estimated Return:', str(round(100*optimal_return_volatility, 3))+'%')
        print('Estimated Volatility:', round(optimal_stdev_volatility, 3))
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

        print('Portfolio Allocation (Maximising Sharpe):')
        for idx, weight in enumerate(optimal_weights_sharpe):
            print(self.asset_list[idx]+':', str(round(100*weight, 3))+'%')
        print('-'*20)
        print('Estimated Return:', str(round(100*optimal_return_sharpe, 3))+'%')
        print('Estimated Volatility:', round(optimal_stdev_sharpe, 3))
        print('-'*20)

        # find efficient frontier for returns in range [0.001, 0.14]
        efficient_returns = np.linspace(0.001, 0.14, 50)
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
        min_stdev = np.argmin(efficient_stdevs)
        efficient_returns = efficient_returns[min_stdev:]
        efficient_stdevs = efficient_stdevs[min_stdev:]

        # generate spline curve of efficient frontier
        spline = sci.splrep(efficient_stdevs, efficient_returns)
        efficient_returns = sci.splev(efficient_stdevs, spline)

        # return and volatility when efficient frontier is tangent to risk-free line
        optimal_stdev_tangent = sco.fsolve(
            func = lambda stdev: abs(
                (sci.splev(stdev, spline) - self.risk_free_rate) / stdev - 
                sci.splev(stdev, spline, der=1)
            ), 
            x0 = 1
        )
        optimal_return_tangent = sci.splev(optimal_stdev_tangent, spline)[0]

        # portfolio allocation when efficient frontier is tangent to risk-free line
        portfolio_alloc_tangent = sco.minimize(
            fun = self.portfolio_stdev,
            x0 = [1/n_assets for _ in range(n_assets)], 
            args = returns,
            method = 'SLSQP',
            bounds = [(0, 1) for _ in range(n_assets)],
            constraints = (
                {'type': 'eq', 'fun': lambda x: self.portfolio_return(x, returns) - optimal_return_tangent},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, 
            )
        )
        assert portfolio_alloc_tangent['success'], portfolio_alloc_tangent['message']
        optimal_weights_tangent = portfolio_alloc_tangent['x']
        
        print('Portfolio Allocation (Efficient Frontier with tangent to risk-free line):')
        for idx, weight in enumerate(optimal_weights_tangent):
            print(self.asset_list[idx]+':', str(round(100*weight, 3))+'%')
        print('-'*20)
        print('Estimated Return:', str(round(100*float(optimal_return_tangent), 3))+'%')
        print('Estimated Volatility:', round(float(optimal_stdev_tangent), 3))

        '''
        Calculate client portfolio statistics
        '''

        client_id = self.client_id
        client_weights = self.client_data.iloc[2:]
        client_return = self.portfolio_return(client_weights, returns)
        client_stdev = self.portfolio_stdev(client_weights, returns)

        '''
        PLOTTING MODEL RESULTS
        '''
        
        plt.figure(figsize=(16, 8))
        # plot return and volatility of simulated portfolios
        plt.scatter(
            x = simulated_stdevs, 
            y = simulated_returns*100, 
            c = simulated_returns*100 / simulated_stdevs,
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
            'm*',
            markersize = 15.0, 
            label = 'Highest Sharpe Ratio'
        )
        # plot optimal tangent portfolio (with risk-free asset)
        plt.plot(
            optimal_stdev_tangent, 
            optimal_return_tangent*100, 
            'b*', 
            markersize = 15.0, 
            label = 'Optimal Tangent (with risk-free asset)'
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
        plt.title('Client '+str(self.client_id)+' Portfolio Simulation', {'size': 20})
        plt.legend(loc = 'upper right')
        plt.grid(True)
        plt.xlabel('Expected volatility')
        plt.ylabel('Expected return (%)')
        plt.colorbar(label = 'Sharpe ratio')
        plt.savefig(self.OUTPUT_PATH / Path('optimal_portfolios.png'))
        plt.close()


    def portfolio_return(self, weights, returns):
        try:
            return np.sum(returns.mean() * weights)
        except ValueError:
            raise ValueError('Number of weights and assets must match')
    

    def portfolio_stdev(self, weights, returns):
        try:
            return np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
        except ValueError:
            raise ValueError('Number of weights and assets must match')


    def portfolio_sharpe(self, weights, returns):
        try:
            return self.portfolio_return(weights, returns) / self.portfolio_stdev(weights, returns)
        except ValueError:
            raise ValueError('Number of weights and assets must match')