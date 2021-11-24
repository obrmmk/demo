try:
    import pegtree as pg
except ModuleNotFoundError:
    import os
    os.system('pip install pegtree')
    import pegtree as pg
import re

## 前処理

peg = pg.grammar('nlpcode.pegtree')
parser = pg.generate(peg)

def fix(tree):
    a = [tree.epos_]
    for t in tree:
        a.append(fix(t).epos_)
    for key in tree.keys():
        a.append(fix(tree.get(key)).epos_)
    tree.epos_ = max(a)
    return tree

def replace_special_token(s):
    tree = parser(s)
    ss = []
    vars = {}
    index = 0
    for t in tree:
        tag = t.getTag()
        print(repr(t))
        if tag == 'Chunk' or tag == 'Special':
            ss.append(str(t))
        else:
            key = ('ABCDEFGHIJKLMNOPQRSTUVZXYZ')[index]
            key = f'<{key}>'
            vars[key] = str(fix(t))
            ss.append(key)
            index += 1
    return ''.join(ss), vars

if __name__ == '__main__':
    print(*replace_special_token('Aの度数分布図をビンをBとして描写する'))
    print(*replace_special_token('Aの度数分布図をビンを B として描写する'))
    print(*replace_special_token('<A>の度数分布図をビンを<B>として描写する'))

