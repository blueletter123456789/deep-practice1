from layer_naive import *

apple = 100
apple_amount = 2
tax = 1.1

# Create each layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_amount)
price = mul_tax_layer.forward(apple_price, tax)
print(price)

dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_amount = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_amount, dtax)
