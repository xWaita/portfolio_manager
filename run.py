from portfolio_manager import PortfolioManager
from sentiment_advisor import SentimentAdvisor

def main():
    pm = PortfolioManager()
    pm.ETL()
    pm.feature_engineering()
    pm.model_design()
    pm.model_implementation()

    exit()

    sa = SentimentAdvisor()
    sa.ETL()
    sa.feature_engineering()
    sa.model_implementation()

if __name__ == '__main__':
    main()