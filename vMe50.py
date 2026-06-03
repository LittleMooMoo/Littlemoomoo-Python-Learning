import numpy as np

print("我重生了，上一世我本是家世显赫的富家少爷，却被青梅竹马联合绿茶男陷害， 导致家人弃我，师门逐我，甚至断我筋脉折磨致死！ 如今重来一世我定要让背叛我的人付出代价，夺回属于我的一切！ KFC疯狂星期四v我50请我吃肯德基，聆听我的复仇计划！")

print("v你50")

print("不v")

choice = input('请输入你的选择：')

if choice == 'v你50':
    print('谢谢老板，好人一生平安')
else:
    print('从你的钱包里偷走50，你的钱包HP: -50')

hello_hello = 2


def goo_goo_ga_ga(ding_ding_level):
    dumb_dumb_level = -(np.e ** (-ding_ding_level)) * (np.cos(hello_hello *
                                                              ding_ding_level)+np.sin(hello_hello * ding_ding_level))
    goo_goo_ga_ga_level = -np.log(np.abs(dumb_dumb_level))
    return goo_goo_ga_ga_level, dumb_dumb_level


ding_ding_level = int(input('请输入moo moo的呆呆值:'))
goo_goo_ga_ga_level, dumb_dumb_level = goo_goo_ga_ga(ding_ding_level)
print(f'moo moo的呆呆值是{ding_ding_level}, 那么ding ding的呆呆值是{dumb_dumb_level}')
print(
    f'moo moo的呆呆值是{ding_ding_level}, 那么goo goo ga ga的level是{goo_goo_ga_ga_level}')

question = input("陪我玩哞哞游戏吗? (yes/no) ")
if question == "yes":
    happy_level = (np.abs(goo_goo_ga_ga_level)) ** 2
    print(f"哞哞happy level: {happy_level}")
else:
    happy_level = goo_goo_ga_ga_level / 2
    print(f"哞哞happy level: {happy_level}")
