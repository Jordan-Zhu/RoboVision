newDict = {"bruh": [100,200,100], "toobruh": 200}
newDict2 = {"threebruh": newDict["toobruh"] + 100}
newDict.update(newDict2)
print(newDict["bruh"])
print(parameters)