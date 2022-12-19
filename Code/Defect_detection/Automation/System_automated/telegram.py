from MESConnection import MESConnection
# If we import only MesConnection doesn't work -> we need to import class MESConnection inside MESConnection file
import socket
import time
import datetime

header = MESConnection.Header("-1")
locationHeader = MESConnection.LocationHeader("8080", "30", "1", "1", "1", "1", "1260", "Defect_detection", "PC")
# resultHeader = MESConnection.ResultHeader("1", msg)  # msg was not defined
resultHeader = MESConnection.ResultHeader("1", "999999999", workingCode="1")  # 1 = Good / 2 = Bad , product number

""" 
Constructs the telegram to be sent
Uses a header, locationHeader and resultHeader instances
Uses the identifier set beforehand. If no identifier was set, it is assigned the value "test"
"""

# Build identifier
currentDate = datetime.datetime.now()
year = str(currentDate.year)
month = str(currentDate.month)
day = str(currentDate.day)
hour = str(currentDate.hour)
min = str(currentDate.minute)
sec = str(currentDate.second)
identifier = '8370_' + year + '_' + month + '_' + day + '_' + hour + '_' + min + '_' + sec + '_999999999'

mesConnection = MESConnection(header, locationHeader, resultHeader, identifier,
                              resHeadEnabled=True)

# ============ TEST ADD ARRAY ===================
#
array = MESConnection.customArray('TestInfo')
# array.addItem(name="Riscos", value="90", unit="%")
array = array.addItems()

# Create telegram
# message = mesConnection.CreateTelegram(array=array)
message = mesConnection.CreateTelegram(array=array)

# Transform the message into an array of bytes
telegramBytes = mesConnection.BuildTelegram(message)

# Create socket -> s wasn't defined
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
target_port = 55065
# target_port = 55765
# target_host = "ims.mec.ua.pt"
target_host = "localhost"
s.connect((target_host, target_port))

s.send(telegramBytes)

# MES response
# Remove the first four bytes (they are just the size of the message)

data = s.recv(1024).decode(encoding='utf-8', errors='ignore')
# print(data)

telegramResult = mesConnection.ResultTelegram().ProcessResponse(data)
print(telegramResult)

time.sleep(1) 

s.close() 

