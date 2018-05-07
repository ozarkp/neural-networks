#!/usr/bin/env python3
"""
vsatest.py : Test code for our VSA class
"""

from vsa import VSA

print('\nPart 2 -----------------------------')

vsa = VSA(1000)

a = vsa.randvec('A')
b = vsa.randvec('B')
c = vsa.randvec('C')
d = vsa.randvec('D')
x = a*b + c*d
y = c*x

print('\nWinner is ' + vsa.winner(y) + ' (should be D)')

print('\nPart 3 -----------------------------')

vsa = VSA(10000) # Ten thousand elements, as per Kanerva (2008)
nam = vsa.randvec('name')
usa = vsa.randvec('USA')
cap = vsa.randvec('capital')
wdc = vsa.randvec('Washington, DC')
mon = vsa.randvec('money')
dol = vsa.randvec('dollar')
mex = vsa.randvec('Mexico')
mxc = vsa.randvec('Mexico City')
pes = vsa.randvec('peso')
ustates = nam*usa + cap*wdc + mon*dol
mexico  = nam*mex + cap*mxc + mon*pes
fum = ustates * mexico
query = dol*fum
print('\nWinner is ' + vsa.winner(query) + ' (should be peso)')

print('\nPart 5 -----------------------------')

vsa = VSA(10000)

a = vsa.randvec('A')
b = vsa.randvec('B')
c = vsa.randvec('C')
d = vsa.randvec('D')
e = vsa.randvec('E')
f = vsa.randvec('F')
g = vsa.randvec('G')

x = vsa.seqencode(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
print('\nDecoded sequence is ' + str(vsa.seqdecode(x)) + ' (should be [A, B, C, D, E, F, G])')

x = vsa.seqencode(['A', 'C', 'E'])
print('\nDecoded sequence is ' + str(vsa.seqdecode(x)) + ' (should be [A, C, E])')
