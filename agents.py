

class CreateAgent:
    def __init__(self, type, id):
        self.type = type
        self.id = id

        if type == 0: # UAV
            self.health = 1
            self.firepower = 1
            self.eff = [1, 0, 1] # Effectiveness layer [ground, air, sea]
            print("UAV-{} is Created".format(self.id))

        if type == 1: # UGV
            self.health = 1
            self.firepower = 1
            self.eff = [1, 1, 1] # Effectiveness layer [ground, air, sea]
            print("UGV-{} is Created".format(self.id))

        if type == 2: # USV
            self.health = 1
            self.firepower = 1
            self.eff = [1, 0, 1] # Effectiveness layer [ground, air, sea]
            print("USV-{} is Created".format(self.id))




