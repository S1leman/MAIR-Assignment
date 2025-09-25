from state_transition import RestaurantSystem

def main():
    print("Initializing the System...")
    system = RestaurantSystem()

    # Configurability: give user option to choose between baseline or ML model
    print("\033[1m" + "CONIFIGURABILITY" + "\033[0m")
    print("Choose a classifier:")
    print("1 - Machine Learning (MLP)")
    print("2 - Majority Baseline")
    print("3 - Rules Baseline")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        system.classifier_type = "mlp"
    elif choice == "2":
        system.classifier_type = "majority"
    elif choice == "3":
        system.classifier_type = "rules"
    else:
        print("Invalid choice, defaulting to: MLP")
        system.classifier_type = "mlp"

    
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
