from state_transition import RestaurantSystem
from utils_colors import create_color_formatter, Colors


def main():
    """
    Main entry point for the Cambridge Restaurant Recommendation System.
    Handles system configuration, initialization, and conversation execution.
    
    Input: None
    Output: None - Runs interactive dialog system until user exits
    """ 
    import sys
    term_width = 100
    print("\n" + "="*term_width)
    print("CAMBRIDGE RESTAURANT RECOMMENDATION SYSTEM".center(term_width))
    print("="*term_width)

    system = RestaurantSystem()

    # Check for -default flag
    use_default = "-default" in sys.argv

    if use_default:
        # Set defaults: no colors, normal letters, allow restarts
        use_colors = False
        system.color_formatter = create_color_formatter(use_colors)
        system.classifier_type = "mlp"
        system.allow_restarts = True
        system.output_caps = False
        print("\nRunning with default configuration: plain text, normal letters, restarts allowed.")
    else:
        print("\n" + "-"*term_width)
        print("SYSTEM CONFIGURATION".center(term_width))
        print("-"*term_width)

        # Configure color interface
        print("\nHow would you like your interface to look?")
        print("  1 - Colored (modern, visually enhanced)")
        print("  2 - Plain text (simple, classic)")
        color_choice = input("\nYour choice (1 for colored, 2 for plain) [default: 1]: ").strip()
        use_colors = color_choice != "2"
        system.color_formatter = create_color_formatter(use_colors)
        if use_colors:
            print("Great! You'll enjoy a colorful, modern experience.")
        else:
            print("Classic mode selected. Simple and distraction-free!")

        # Configure classifier
        system.classifier_type = "mlp"

        # Configure restart policy
        print("\nWould you like to be able to restart the conversation at any time?")
        print("  1 - Yes, allow restarts (more flexible)")
        print("  2 - No, keep it linear (one-way flow)")
        restart_choice = input("\nYour choice (1 for restarts, 2 for linear) [default: 1]: ").strip()
        system.allow_restarts = restart_choice != "2"
        if system.allow_restarts:
            print("Restarts are enabled. You can start over whenever you like!")
        else:
            print("Linear flow selected. The conversation will move forward step by step.")

        # Configure output format
        print("\nHow do you want system messages to appear?")
        print("  1 - Normal capitalization (easy to read)")
        print("  2 - ALL CAPS (for extra emphasis)")
        caps_choice = input("\nYour choice (1 for normal, 2 for ALL CAPS) [default: 1]: ").strip()
        system.output_caps = caps_choice == "2"
        if system.output_caps:
            print("ALL CAPS mode enabled. System messages will be loud and clear!")
        else:
            print("Normal case selected. Messages will be easy on the eyes.")
    
    # Load ML model
    system.ensure_model_ready()
    
    # Clear screen and show configured system info
    print("\n" + "-"*term_width)
    
    if use_colors:
        print(system.color_formatter.success_message("System Ready!", caps=system.output_caps))
        print(system.color_formatter.header("NAVIGATION GUIDE", width=term_width, caps=system.output_caps))

        navigation_info = [
            "• Type 'exit', 'quit', or 'bye' to leave anytime",
            "• Press Ctrl+C for emergency exit"
        ]

        if system.allow_restarts:
            navigation_info.insert(1, "• Type 'start over' or 'restart' to begin again")

        if system.output_caps:
            navigation_info.append("• System responses will be in ALL CAPS")

        for info in navigation_info:
            print(system.color_formatter.info_message(info, caps=system.output_caps).center(term_width))
    else:
        print("System Ready!".center(term_width))
        print("-"*term_width)
        print("\nNavigation:".center(term_width))
        print("Type 'exit', 'quit', or 'bye' to leave anytime".center(term_width))
        if system.allow_restarts:
            print("Type 'start over' or 'restart' to begin again".center(term_width))
        print("Press Ctrl+C for emergency exit".center(term_width))
        if system.output_caps:
            print("System responses will be in ALL CAPS".center(term_width))
        print("\n" + "="*term_width + "\n")
     
    try:
        # Pass term_width to conversation header/footer/turns for consistent width
        system._print_conversation_header(term_width=term_width)
        while system.current_state and not system.conversation_ended:
            system._handle_conversation_turn(term_width=term_width)
        system._print_conversation_footer(term_width=term_width)
    except KeyboardInterrupt:
        # Handle Ctrl+C shutdown
        if use_colors:
            print("\n")
            msg = "Thank you for using the Cambridge Restaurant System!"
            if system.output_caps:
                msg = msg.upper()
            print(system.color_formatter.success_message(msg, caps=system.output_caps))
        else:
            print("\n\nThank you for using the Cambridge Restaurant System!")
            print("Have a great day!\n")
    except Exception as e: 
        if use_colors:
            print(system.color_formatter.error_message(f"Error during conversation: {e}", caps=system.output_caps))
        else:
            print(f"\nError during conversation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()