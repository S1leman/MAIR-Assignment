from state_transition import RestaurantSystem

def main():
    print("Initializing the System...")
    system = RestaurantSystem()
    
    system.ensure_model_ready()
       
    num_restaurants = len(system.restaurant_lookup.df)
    
    print()
    print("(Use Ctrl+C to exit)")
    
    try:
        system.run_conversation()
    except KeyboardInterrupt:
        print("Thank you for using the Cambridge Restaurant System!")
    except Exception as e:
        print(f"\nError during conversation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
