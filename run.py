from portfolio_manager import PortfolioManager
from sentiment_advisor import SentimentAdvisor

def main():
    pm = PortfolioManager()
    pm.ETL()
    pm.feature_engineering()
    pm.model_design()

    sa = SentimentAdvisor()
    sa.ETL()

if __name__ == '__main__':
    main()