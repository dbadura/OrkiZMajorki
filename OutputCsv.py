class Output():
    def __init__(self, train_number, left_right):
        self.team_name = 'Orki z majorki'
        self.train_number = 1
        self.left_right = 'left_right'
        self.frame_number = 0
        self.wagon = 0
        self.uic_0_1 = '0'
        self.uic_label = '0'

    def __str__(self):
        output = self.team_name + ','
        output += str(self.train_number) + ','
        output += self.left_right + ','
        output += str(self.frame_number) + ','
        output += str(self.wagon) + ','
        output += self.uic_0_1 + ','
        output += str(self.uic_label)
        return str(output)
