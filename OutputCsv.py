class Output():
    def __init__(self, train_number, left_right):
        self.team_name = 'Orki z majorki'
        self.train_number = train_number
        self.left_right = 'left_right'
        self.frame_number = 0
        self.wagon = 0
        self.uic_0_1 = '0'
        self.uic_label = '0'

    def __str__(self):
        output = self.team_name + ','
        output += str(self.train_number) + ','
        output += self.left_right + ','
        self.frame_number = str(self.frame_number).lstrip('0')
        if self.frame_number=='':
            self.frame_number='0'
        output += str(self.frame_number) + ','
        output += str(self.wagon) + ','
        output += str(self.uic_0_1) + ','
        output += str(self.uic_label)
        return str(output)

    def compare(self, other):
        diff = ''
        if self.train_number !=other.train_number:
            diff+= ' train number:'+self.train_number +'!=' +other.train_number

        if self.left_right !=other.left_right:
            diff+= ' left_right: '+ self.left_right +'!=' +other.left_right

        if self.frame_number !=other.frame_number:
            diff+= ' frame_number: '+  self.frame_number +'!=' +other.frame_number

        if self.train_number !=other.train_number:
            diff+= ' train_number: ' + self.train_number +'!=' +other.train_number

        if self.wagon !=other.wagon:
            diff+= ' wagon: ' + self.wagon +'!=' +other.wagon

        if self.uic_0_1 !=other.uic_0_1:
            diff+= ' uic_0_1: '+ self.uic_0_1 +'!=' +other.uic_0_1

        if self.uic_label !=other.uic_label:
            diff+= ' uic_label: ' + self.uic_label +'!=' +other.uic_label

        if diff=='':
            diff='OK'
        return diff


