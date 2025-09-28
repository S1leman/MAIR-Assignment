from state_transition import RestaurantSystem

def main():
    # display welcome header and initialize system
    print("\n" + "="*60)
    print("CAMBRIDGE RESTAURANT RECOMMENDATION SYSTEM")
    print("="*60)
    print("\nInitializing the System...")
    system = RestaurantSystem()

    print("\n" + "-"*40)
    print("SYSTEM CONFIGURATION")
    print("-"*40)
    
    # get classifier choice from user
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
    
    # configure restart policy
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
    
    # configure output format
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
    
    # configure text-to-speech
    print("\nText-to-Speech:")
    print("  1 - Disabled - Text output only")
    print("  2 - Enabled - System will speak responses aloud")
    
    tts_choice = input("\nEnter your choice (1/2) [default: 1]: ").strip()
    
    if tts_choice == "2":
        try:
            system.use_tts = True
            system.initialize_tts()
            print("  Text-to-speech ENABLED")
        except ImportError:
            print("  WARNING: pyttsx3 not installed. TTS disabled.")
            print("  Install with: pip install pyttsx3")
            system.use_tts = False
        except Exception as e:
            print(f"  WARNING: TTS initialization failed: {e}")
            print("  Continuing with text-only output")
            system.use_tts = False
    else:
        system.use_tts = False
        print("  Text-to-speech disabled")
    
    # load or train the classifier model
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
    if system.use_tts:
        print("  - System will speak responses aloud")
    
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
    finally:
        # cleanup TTS engine if it was initialized
        if hasattr(system, 'tts_engine') and system.tts_engine:
            try:
                system.tts_engine.stop()
            except:
                pass

if __name__ == "__main__":
    main()