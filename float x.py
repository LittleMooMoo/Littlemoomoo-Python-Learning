Good_ding_ding_level = 250000


def moo_moo(ding_ding_level):
    if ding_ding_level < Good_ding_ding_level:
        return "hungry"
    elif ding_ding_level > Good_ding_ding_level:
        return "sleepy"
    else:
        return "I am little moo moo"


user_input = input("what did you eat today? moomoo:")

try:

    ding_ding_level = float(user_input)
    result = moo_moo(ding_ding_level)
    print(result)
except ValueError:
    print("Goo Goo Gaa Gaa")
