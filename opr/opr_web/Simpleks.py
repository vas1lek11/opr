from copy import deepcopy


class Simpleks:
    def __init__(self, price1, price2, cost, req_area1, req_area2, area, out_put1, out_put2):
        self.simp_table = [[cost, price1, price2, 1, 0],
                           [area, req_area1, req_area2, 0, 1],
                           [0, -out_put1, -out_put2, 0, 0]]
        for i in range(len(self.simp_table)):
            for j in range(len(self.simp_table[i])):
                print((self.simp_table[i][j]), end=" ")
            print()
        self.basic_row = ["x3", "x4"]
        self.basic_col = ["x1", "x2", "x3", "x4"]

    def findMainCol(self):
        mainCol = 1
        m = len(self.simp_table)
        n = len(self.simp_table[m-1])
        for j in range(2,n):
            if self.simp_table[m - 1][j] < self.simp_table[m - 1][mainCol]:
                mainCol = j
        #print(mainCol)
        return mainCol

    def findMainRow(self, maincol):
        mainrow = 0
        m = len(self.simp_table)
        n = len(self.simp_table[m - 1])
        for i in range(0, m-1):
            if self.simp_table[i][maincol] > 0:
                mainrow = i
                break
        for i in range(mainrow+1, m-1):
            if ((self.simp_table[i][maincol] > 0) and
                    ((self.simp_table[i][0] / self.simp_table[i][maincol]) < (self.simp_table[mainrow][0] / self.simp_table[mainrow][maincol]))):
                mainrow = i
        self.basic_row[mainrow] = self.basic_col[maincol - 1]
        return mainrow

    def IsItEnd(self):
        flag = False
        m = len(self.simp_table)
        n = len(self.simp_table[m - 1])
        for i in range(1, n):
            if self.simp_table[m-1][i] < 0:
                flag = True
                break
        return flag

    def Calculate(self):
        count = 1
        new_table = deepcopy(self.simp_table)
        #for i in range(len(self.simp_table)):
            #for j in range(len(self.simp_table[i])):
                #new_table[i][j] = self.simp_table[i][j]
            #print()
        for i in range(len(self.simp_table)):
            for j in range(len(self.simp_table[i])):
                print((new_table[i][j]), end=" ")
            print()
        while self.IsItEnd():
            maincol = self.findMainCol()
            mainrow = self.findMainRow(maincol)
            m = len(self.simp_table)
            n = len(self.simp_table[m - 1])
            print(m, "   ", n, "    \n")
            k = self.simp_table[mainrow][maincol]


            for j in range(0, n):
                new_table[mainrow][j] = self.simp_table[mainrow][j]/k


            for i in range(0, m):
                if i == mainrow:
                    continue
                for j in range(0, n):
                    zam = self.simp_table[i][j]
                    zam2 = self.simp_table[i][maincol]
                    zam3 = self.simp_table[mainrow][j]
                    new_table[i][j] = zam - (zam2 * zam3)/k
            print(count, "этап решения:\n")
            for i in range(len(self.simp_table)):
                for j in range(len(self.simp_table[i])):
                    print((self.simp_table[i][j]), end=" ")
                print()
            count = count + 1
            self.simp_table = deepcopy(new_table)
        print("Конечная таблица:\n")
        for i in range(len(self.simp_table)):
            for j in range(len(self.simp_table[i])):
                print((self.simp_table[i][j]), end=" ")
            print()
        print(self.basic_row[0], "=", self.simp_table[0][0], "\n")
        print(self.basic_row[1], "=", self.simp_table[1][0], "\n")

if __name__ == '__main__':
    pr = Simpleks(2, 4, 100, -0.4, 0.6, 0., 20, 40)
    pr.Calculate()

