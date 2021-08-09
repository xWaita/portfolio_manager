from portfolio_optimiser import PortfolioOptimiser
from sentiment_advisor import SentimentAdvisor

def run_portfolio_optimiser():
    print_title(' RUNNING PORTFOLIO OPTIMISER ')

    pm = PortfolioOptimiser()
    pm.ETL()
    pm.feature_engineering()
    pm.model_design()
    pm.model_implementation()

def run_sentiment_advisor():
    print()
    print_title(' RUNNING SENTIMENT ADVISOR ')

    sa = SentimentAdvisor()
    sa.ETL()
    sa.feature_engineering()
    sa.model_design()
    sa.model_implementation()

def print_title(title: str):
    print('#'*(len(title)+20))
    print('#'*10+title+'#'*10)
    print('#'*(len(title)+20))
    print()

def main():
    run_portfolio_optimiser()
    # run_sentiment_advisor()

if __name__ == '__main__':
    main()