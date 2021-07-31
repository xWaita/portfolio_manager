from portfolio_manager import PortfolioManager

def main():
    pm = PortfolioManager()
    pm.ETL()
    pm.feature_engineering()
    pm.model_design()

if __name__ == '__main__':
    main()