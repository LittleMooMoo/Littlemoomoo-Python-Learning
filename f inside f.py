def moo_moo_mood(ding_ding_level):
    def hungry_moo():
        return ding_ding_level <= 25
    if hungry_moo():
        return ("happy cow")
    else:
        return ("sad cow")


user_input = input("please type in your ding_ding_level:")
ding_ding_level = float(user_input)

result = moo_moo_mood(ding_ding_level)
print(result)
