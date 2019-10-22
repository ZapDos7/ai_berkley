class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

#    def isFull(self):
#        return self.items != []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

#INPUT = string with ()[]{}, zugismenoi or not - USE STACK

def Match(s1, s2):
    if s1 == '[':
        return s2 == ']'
    elif s1 == '{':
        return s2 == '}'
    elif s1 == '(':
        return s2 == ')'
    else:
        return False

def Check(symbolString):
    
    s = Stack() #we start with an empty stack
    for symbol in symbolString:

        if symbol == '[' or symbol == '{' or symbol == '(':
            s.push(symbol)
        elif symbol == ']' or symbol == '}' or symbol == ')':
            if s.isEmpty() == True:
                print "not ok!"
                return
            else:
                temp = s.pop()
                if Match(temp,symbol) == False:
                    print "not ok!"
                    return
        else:
            print "invalid character!"
            return
    if s.isEmpty() == True:
        print "ok!"
    else:
        print "not ok!"
        #else dont print anything
"""
print(Check('((()))')) #ok
print(Check('(()')) #not ok
print(Check('{[]}')) #ok
print(Check('{()]}[')) #not ok
print(Check('(a)')) #invalid
print(Check('{[}]')) #NOT ok """