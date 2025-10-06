import pandas as pd

class RestaurantLookup:
    def __init__(self, csv_path: str = "data/restaurant_info.csv"):
        """
        Initialize restaurant lookup with CSV data.
        
        Input: csv_path (string) - path to restaurant CSV file
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)
        # Replace NaN with None
        self.df = self.df.where(pd.notnull(self.df), None)

    def lookup(self, filters):
        """
        Find restaurants matching user preferences.
        
        Input: filters (dict) - user preferences {"food": "italian", "area": "north", "price": "cheap"}
        Output: (restaurant_name, alternatives_list) tuple OR (None, []) if no matches
        """
        if self.df is None:
            return None, []
        
        filtered_df = self.df.copy()
        
        # Apply each filter unless user doesn't care
        for key, value in filters.items():
            if value != "dontcare":
                if key in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[key].str.lower() == value.lower()]
        
        if filtered_df.empty:
            return None, []
        
        first_match = filtered_df.iloc[0].to_dict()
        
        # Get remaining matches as alternatives
        alternatives = filtered_df.iloc[1:].to_dict(orient="records")
        
        return first_match["restaurantname"], alternatives
    
