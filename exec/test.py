class Atest():
    def __init__(self, vb):
        self.va = 1
        self.vb = vb

    def __call__(self):
        self.vc = self.va + self.vb

vB = 2

atest = Atest(vB)
atestcase = atest()

print(atest.va)
print(atest.vb)
print(atest.vc)