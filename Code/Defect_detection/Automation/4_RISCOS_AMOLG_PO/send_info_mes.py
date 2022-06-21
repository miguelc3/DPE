import socket
import sys

# Create a connection to the server application on port 55965
tcp_socket = socket.create_connection(('ims.mec.ua.pt', 55965))

# ____________ READ XML _________________
with open('example_PartProcessed_nospaces.xml', 'r') as f:
    xml_data = f.read()

# ____________ PROCESS XML _________________
xml_size_og = len(xml_data)  # original size
xml_size = len(xml_data) + 4  # add 4 bytes (MES needs these to know xml size)
print('\nSize = ' + str(xml_size_og))

# Creating 4 header bytes
lolo = (xml_size & 0xff)  # 0xff in hex -> 255 in dec
hilo = ((xml_size >> 8) & 0xff)
lohi = ((xml_size >> 16) & 0xff)
hihi = (xml_size >> 24)

# Array of decimals
telegram_bytes = [hihi, lohi, hilo, lolo] # save values in array
print("\n")
print(telegram_bytes)

# Array of bytes to be sent to MES
telegram_bytes = bytearray(telegram_bytes)  # convert normal array (decimals) into an

print("\n")
print(telegram_bytes)

# Cyclically read character from xml, convert to decimal, append to array of bytes
for i in range(0, xml_size_og):
    char_to_decimal = ord(xml_data[i])
    telegram_bytes.append(char_to_decimal)


try:
    tcp_socket.sendall(telegram_bytes)
    print("\n")
    print('Sent: ' + str(telegram_bytes))

finally:
    print("Closing socket")
    tcp_socket.close()




