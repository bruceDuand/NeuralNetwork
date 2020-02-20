filter1 = "AAB3289758A4492E4D9DFD15B438F1A6"
filter2 = "AAB3289758A4492E4D9DFD15B438F1A6"
filter3 = "33D372F60B238D934725A60E9F8B625C"
filter4 = "7B9BDD17979B81A2A92270C95DBBD95B"

B = "7B9BDD17979B81A2A92270C95DBBD95B"
b1 = "5B"
print(hex(int("F1", 16)*int("D9", 16)))

batch_size = 2
filter1_hex_code = [filter1[k:k+batch_size] for k in range(0, len(filter1), batch_size)]
B_hex_code = [B[k:k+batch_size] for k in range(0, len(B), batch_size)]
l = len(B_hex_code)
print(l)

ml = l
res_hex = hex(sum([int(f, 16)*int(b, 16) for f, b in zip(filter1_hex_code[l-ml:], B_hex_code[l-ml:])]))
print(res_hex)

signed_adder_code = ["04A392", "03AC82", "03AC82", "03FFB4"]
signed_adder_sum  = hex(sum([int(k, 16) for k in signed_adder_code]))
print(signed_adder_sum)