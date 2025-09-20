import pandas as pd

class RestaurantLookup:
    
    def __init__(self, csv_path: str = "data/restaurant_info.csv"):
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)

    def lookup(self, filters):
        if self.df is None:
            return None, []
        
        filtered_df = self.df.copy()
        
        for key, value in filters.items():
            if value != "dontcare":
                if key in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[key].str.lower() == value.lower()]
        
        if filtered_df.empty:
            return None, []
        
        first_match = filtered_df.iloc[0].to_dict()
        
        alternatives = filtered_df.iloc[1:].to_dict(orient="records")
        
        return first_match["restaurantname"], alternatives
    
