try:
    import pegtree as pg
except ModuleNotFoundError:
    import os
    os.system('pip install pegtree')
    import pegtree as pg
import re

# 前処理

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


def replace_expression(s):
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


def replace_special(s, vars):
    for key in vars:
        s = s.replace(key, vars[key])
    return s


def compose_nmt(nmt, replace_before=replace_expression, replace_after=replace_special):
    def translate(s, beams=5):
        s, vars = replace_before(s)
        pred, prob = nmt(s, max(beams, 5))
        pred = [replace_after(s, vars) for s in pred]
        if beams <= 1:
            return pred[0]
        return pred, prob
    return translate


if __name__ == '__main__':
    print(*replace_expression('Aの度数分布図をビンをBとして描写する'))
    print(*replace_expression('Aの度数分布図をビンを B として描写する'))
    print(*replace_expression('<A>の度数分布図をビンを<B>として描写する'))
