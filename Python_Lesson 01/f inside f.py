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

user_input = input("please enter casual moo moo value:")
casual_moo_moo_value = float(user_input)


def moo_moo_cool_level(casual_moo_moo_value, ding_ding_level):
    return max(casual_moo_moo_value, ding_ding_level)


def moo_moo_overall_mood(casual_moo_moo_value, ding_ding_level):
    moo_level = moo_moo_cool_level(casual_moo_moo_value, ding_ding_level)

    if moo_level < 909:
        return ("sad moo")
    else:
        return ("happy moo")


result = moo_moo_overall_mood(casual_moo_moo_value, ding_ding_level)
print(result)
