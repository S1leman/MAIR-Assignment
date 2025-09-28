from state_transition import RestaurantSystem

def main():
    print("\n" + "="*60)
    print("CAMBRIDGE RESTAURANT RECOMMENDATION SYSTEM")
    print("="*60)
    print("\nInitializing the System...")
    system = RestaurantSystem()

    # Configuration Menu
    print("\n" + "-"*40)
    print("SYSTEM CONFIGURATION")
    print("-"*40)
    
    # Classifier configuration
    print("\nChoose Classifier Type:")
    print("  1 - Machine Learning (MLP) - Most accurate")
    print("  2 - Majority Baseline - Simple baseline")
    print("  3 - Rules Baseline - Rule-based approach")
    
    choice = input("\nEnter your choice (1/2/3) [default: 1]: ").strip()
    
    if choice == "2":
        system.classifier_type = "majority"
        print("  Using Majority Baseline classifier")
    elif choice == "3":
        system.classifier_type = "rules"
        print("  Using Rules Baseline classifier")
    else:
        system.classifier_type = "mlp"
        print("  Using Machine Learning (MLP) classifier")
    
    # Restart configuration
    print("\nDialog Restart Policy:")
    print("  1 - Allow restarts - Users can start over anytime")
    print("  2 - No restarts - Linear conversation flow only")
    
    restart_choice = input("\nEnter your choice (1/2) [default: 1]: ").strip()
    
    if restart_choice == "2":
        system.allow_restarts = False
        print("  Restarts disabled - Linear flow enabled")
    else:
        system.allow_restarts = True
        print("  Restarts enabled - Flexible conversation")
    
    system.ensure_model_ready()
    
    print("\n" + "-"*40)
    print("System Ready!")
    print("-"*40)
    print("\nNavigation:")
    print("  - Type 'exit', 'quit', or 'bye' to leave anytime")
    if system.allow_restarts:
        print("  - Type 'start over' or 'restart' to begin again")
    print("  - Press Ctrl+C for emergency exit")
    print("\n" + "="*60 + "\n")
    
    try:
        system.run_conversation()
    except KeyboardInterrupt:
        print("\n\nThank you for using the Cambridge Restaurant System!")
        print("Have a great day!\n")
    except Exception as e:
        print(f"\nError during conversation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()