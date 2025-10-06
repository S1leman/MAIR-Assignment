from state_transition import RestaurantSystem


def main():
    """
    Main entry point for the Cambridge Restaurant Recommendation System.
    Handles system configuration, initialization, and conversation execution.
    
    Input: None
    Output: None - Runs interactive dialog system until user exits
    """ 
    print("\n" + "="*60)
    print("CAMBRIDGE RESTAURANT RECOMMENDATION SYSTEM")
    print("="*60)
    print("\nInitializing the System...")
    system = RestaurantSystem()

    print("\n" + "-"*40)
    print("SYSTEM CONFIGURATION")
    print("-"*40)
    
    # Configure classifier type
    print("\nChoose Classifier Type:")
    print("  1 - Machine Learning (MLP) - Most accurate")
    print("  2 - Majority Baseline - Simple baseline")  
    print("  3 - Rules Baseline - Rule-based approach")
    
    choice = input("\nEnter your choice (1/2/3) [default: 1]: ").strip()
    
    classifier_options = {
        "2": ("majority", "Majority Baseline"),
        "3": ("rules", "Rules Baseline")
    }
    
    if choice in classifier_options:
        classifier_type, classifier_name = classifier_options[choice]
        system.classifier_type = classifier_type
        print(f"  Using {classifier_name} classifier")
    else:
        system.classifier_type = "mlp"
        print("  Using Machine Learning (MLP) classifier")
    
    # Configure restart policy
    print("\nDialog Restart Policy:")
    print("  1 - Allow restarts - Users can start over anytime")
    print("  2 - No restarts - Linear conversation flow only")
    
    restart_choice = input("\nEnter your choice (1/2) [default: 1]: ").strip()
    
    system.allow_restarts = restart_choice != "2"
    if system.allow_restarts:
        print("  Restarts enabled - Flexible conversation")
    else:
        print("  Restarts disabled - Linear flow enabled")
    
    # Configure output format
    print("\nOutput Format:")
    print("  1 - Normal case - Standard capitalization")
    print("  2 - ALL CAPS - All system output in uppercase")
    
    caps_choice = input("\nEnter your choice (1/2) [default: 1]: ").strip()
    
    system.output_caps = caps_choice == "2"
    if system.output_caps:
        print("  ALL CAPS OUTPUT ENABLED")
    else:
        print("  Normal case output enabled")
    
    system.ensure_model_ready()
    
    print("\n" + "-"*40)
    print("System Ready!")
    print("-"*40)
    print("\nNavigation:")
    print("  - Type 'exit', 'quit', or 'bye' to leave anytime")
    if system.allow_restarts:
        print("  - Type 'start over' or 'restart' to begin again")
    print("  - Press Ctrl+C for emergency exit")
    
    if system.output_caps:
        print("  - System responses will be in ALL CAPS")
    
    print("\n" + "="*60 + "\n")
     
    try:
        system.run_conversation()
    except KeyboardInterrupt:
        # Handle Ctrl+C shutdown
        print("\n\nThank you for using the Cambridge Restaurant System!")
        print("Have a great day!\n")
    except Exception as e: 
        print(f"\nError during conversation: {e}") 
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()