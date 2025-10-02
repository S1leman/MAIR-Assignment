from state_transition import RestaurantSystem


def main():
    """
    The function performs the following operations:
    1. System initialization and welcome display
    2. User configuration (classifier, restart policy, output format)
    3. Model loading/training
    4. Conversation execution with error handling
    5. Resource cleanup
    """ 
    print("\n" + "="*60)
    print("CAMBRIDGE RESTAURANT RECOMMENDATION SYSTEM")
    print("="*60)
    print("\nInitializing the System...")
    system = RestaurantSystem()

    print("\n" + "-"*40)
    print("SYSTEM CONFIGURATION")
    print("-"*40)
    
    # Configure classifier type based on user preference 
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
    
    # Configure conversation flow policy 
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
    
    # Configure output formatting for accessibility/preference needs
    print("\nOutput Format:")
    print("  1 - Normal case - Standard capitalization")
    print("  2 - ALL CAPS - All system output in uppercase")
    
    caps_choice = input("\nEnter your choice (1/2) [default: 1]: ").strip()
    
    if caps_choice == "2":
        system.output_caps = True
        print("  ALL CAPS OUTPUT ENABLED")
    else:
        system.output_caps = False
        print("  Normal case output enabled")
    
    # Initialize machine learning model or load pre-trained weights 
    system.ensure_model_ready()
    
    # Display system readiness and user navigation options
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
        # Handle shutdown on user interrupt (Ctrl+C)
        print("\n\nThank you for using the Cambridge Restaurant System!")
        print("Have a great day!\n")
    except Exception as e: 
        print(f"\nError during conversation: {e}") 
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()