from PyInfiniteTree.Core.TaskStackMachine import StatusMessage

# import modules.PyInfiniteTree.Core.TaskStackMachine

class Integration:
    def __init__(self):
        self.sigma = StatusMessage.RUNNING
        print(f"Initial message is: {self.sigma}")
        print("Initial message is RUNNING: ",  self.sigma == StatusMessage.RUNNING)
        print("Initial message is SUCCESS: ", self.sigma == StatusMessage.SUCCESS)

test = Integration()