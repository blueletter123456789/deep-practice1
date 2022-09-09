from layer_naive import *

apple, orange = 100, 150
apple_num, orange_num = 2, 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)

all_price = add_apple_orange_layer.forward(apple_price, orange_price)

price = mul_tax_layer.forward(all_price, tax)

print(apple_price, orange_price, all_price, price)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)

dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)

dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

print(dapple, dapple_num)
print(dorange, dorange_num)
print(dall_price, dtax)
