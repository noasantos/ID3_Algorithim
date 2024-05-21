from ID3_RUNMODES import run
from ID3_VIZUALIZATION import buildTreeImg
import os

def choose_dataframe():
    print("Hello, you can run a simulation now. Shall we begin?")
    print("Choose a dataframe:")
    print("1. iris.csv")
    print("2. restaurant.csv")
    print("3. weather.csv")
    
    df_choice = input("Enter the number of your choice: ")
    if df_choice == '1':
        return 'datasets/iris.csv'
    elif df_choice == '2':
        return 'datasets/restaurant.csv'
    elif df_choice == '3':
        return 'datasets/weather.csv'
    else:
        print("Invalid choice. Please try again.")
        return choose_dataframe()

def choose_mode():
    print("Choose the mode you want:")
    print("1. With all data")
    print("2. With train and test split")
    print("3. With k-fold")
    
    mode_choice = input("Enter the number of your choice: ")
    if mode_choice == '1':
        return 'ALLDATA'
    elif mode_choice == '2':
        return 'TRAINTEST'
    elif mode_choice == '3':
        return 'KFOLD'
    else:
        print("Invalid choice. Please try again.")
        return choose_mode()

def view_decicion_tree():
    print("Would you like to open the generated image in your browser automatically?")
    print("Otherwise you can open it later manually.")
    print("1. Yes")
    print("2. No")
    view = input("Enter the number of your choice: ")
    if view == "1":
        return True
    elif view == "0":
        return False
    else:
        print("Invalid choice. Please try again.")
        return view_decicion_tree()

def main():
    datasource = choose_dataframe()
    mode = choose_mode()
    file_name = os.path.splitext(os.path.basename(datasource))[0]
    
    print(f"You have chosen the dataframe: {file_name}")
    print(f"And the mode: {mode}")
    print("Constructing the decision tree...")
    rootNode = run(datasource, mode)
    if mode in ["ALLDATA", "TRAINTEST"]:
        print("Generating the decision tree graph for this simulation...1")
        view = view_decicion_tree()
        buildTreeImg(rootNode, file_name, mode, view)
    
    try_again = input("We are done. Would you like to run another simulation? (y/n): ")
    if try_again == "y":
        main()
        
    print("Thank you for using our simulatir. See you!")
    
if __name__ == "__main__":
    main()

