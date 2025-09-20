from state_transition import RestaurantSystem

def main():
    print("=" * 70)
    print("Cambridge Restaurant System")
    print("=" * 70)
    print()
    
    print("Initializing the System...")
    system = RestaurantSystem()
    
    print("Training MLP classifier...")
    system.train_classifier()
       
    num_restaurants = len(system.restaurant_lookup.df)
    
    print()
    print("(Use Ctrl+C to exit)")
    print()
    
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
