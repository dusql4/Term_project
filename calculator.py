# Calculator
def calculate(first_number, second_number, operator):
    # Addition (+)
    if(operator == '+'):
        return first_number + second_number
    # Substraction (-)
    elif(operator == '-'):
        return first_number - second_number
    # Multiplication (*)
    elif(operator == '*'):
        return first_number * second_number
    # Division (/)
    elif(operator == '/'):
        return first_number / second_number
    # Power 2 (**)
    elif(operator == '^'):
        return first_number ** second_number
    # Unknown
    else:
        return 'Sorry, but I cannot understand your operation'

# initialization
def init():
    operator = input('Please add the math operator (+, -, *, /, ^)? ')

    print(first_number, operator, second_number, ' = ', calculate(first_number, second_number, operator))
init()
