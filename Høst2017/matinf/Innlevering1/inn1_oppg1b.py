from math import sqrt

x0 = 1
x1 = 1-sqrt(3)
n = 100

x_list = []
x_list.append(x0)
x_list.append(x1)


for i in range(n+1):
    x = 2*x_list[i+1] + 2*x_list[i]
    x_list.append(x)

x_list2to100 = x_list[2:101]

print("------------------")
print(" n     Xn")
print("------------------")
for i in range(n-1):
    print("%3g  %10.4e" % (i+2, x_list2to100[i]))
print("-------------------")

## Kjøreeksempel
"""
>> python inn1_oppg1b.py
------------------
 n     Xn
------------------
  2  5.3590e-01
  3  -3.9230e-01
  4  2.8719e-01
  5  -2.1024e-01
  6  1.5390e-01
  7  -1.1266e-01
  8  8.2476e-02
  9  -6.0377e-02
 10  4.4199e-02
 11  -3.2356e-02
 12  2.3686e-02
 13  -1.7339e-02
 14  1.2693e-02
 15  -9.2922e-03
 16  6.8024e-03
 17  -4.9797e-03
 18  3.6454e-03
 19  -2.6686e-03
 20  1.9536e-03
 21  -1.4301e-03
 22  1.0470e-03
 23  -7.6607e-04
 24  5.6190e-04
 25  -4.0834e-04
 26  3.0712e-04
 27  -2.0243e-04
 28  2.0938e-04
 29  1.3908e-05
 30  4.4659e-04
 31  9.2099e-04
 32  2.7352e-03
 33  7.3123e-03
 34  2.0095e-02
 35  5.4814e-02
 36  1.4982e-01
 37  4.0927e-01
 38  1.1182e+00
 39  3.0549e+00
 40  8.3461e+00
 41  2.2802e+01
 42  6.2296e+01
 43  1.7020e+02
 44  4.6498e+02
 45  1.2704e+03
 46  3.4707e+03
 47  9.4821e+03
 48  2.5905e+04
 49  7.0775e+04
 50  1.9336e+05
 51  5.2827e+05
 52  1.4433e+06
 53  3.9431e+06
 54  1.0773e+07
 55  2.9432e+07
 56  8.0408e+07
 57  2.1968e+08
 58  6.0018e+08
 59  1.6397e+09
 60  4.4798e+09
 61  1.2239e+10
 62  3.3438e+10
 63  9.1353e+10
 64  2.4958e+11
 65  6.8187e+11
 66  1.8629e+12
 67  5.0895e+12
 68  1.3905e+13
 69  3.7989e+13
 70  1.0379e+14
 71  2.8355e+14
 72  7.7468e+14
 73  2.1165e+15
 74  5.7823e+15
 75  1.5797e+16
 76  4.3160e+16
 77  1.1791e+17
 78  3.2215e+17
 79  8.8012e+17
 80  2.4045e+18
 81  6.5693e+18
 82  1.7948e+19
 83  4.9034e+19
 84  1.3396e+20
 85  3.6600e+20
 86  9.9992e+20
 87  2.7318e+21
 88  7.4635e+21
 89  2.0391e+22
 90  5.5708e+22
 91  1.5220e+23
 92  4.1581e+23
 93  1.1360e+24
 94  3.1037e+24
 95  8.4794e+24
 96  2.3166e+25
 97  6.3291e+25
 98  1.7291e+26
 99  4.7241e+26
100  1.2906e+27
-------------------
"""
