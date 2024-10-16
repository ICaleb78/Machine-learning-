pin = input('enter a number:')
try :
    check = int(pin)
except :
    check = -1
if check > 0 :
    print('correct pin')
else :
    print('wrong pin')
