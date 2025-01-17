from A.TaskA import run_A

from B.TaskB import run_B

def main():
        
    #———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # The task occur some unexpected error could not run both task at a time try to commentout one task to run another task.
    #———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    
    run_A()

    run_B()


    

if __name__ == "__main__":
    main()