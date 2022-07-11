# To send data to MES
import socket
# To create a small delay between sending the telegram and closing the socket
import time
# To validate custom timestamps
import datetime
# Parse xml response from MES
import xml.etree.ElementTree as ET


class MESConnection:
    """Python Class that enables you to connect to Nexeed MES and send Telegrams

    Attributes:
        header: An instance of an inner class representing the Telegram Header
        locationHeader: An instance of an inner class representing the location of the station.
        identifier: A string containing the serial number you wish to send to MES.
    """

    def __init__(self, header, locationHeader, BodyTelegram, identifier, arrayItems="", bodyItems="",
                 resHeadEnabled=False):
        """Obtains the information of the three headers
        """
        self.header = header
        self.locationHeader = locationHeader
        self.BodyTelegram = BodyTelegram
        self.identifier = identifier
        self.arrayItems = arrayItems
        self.bodyItems = bodyItems
        self.resHeadEnabled = resHeadEnabled
        self.allItems = ''

    class bodyItemsClass:
        """
        Isolated BodyItems
        structArray


        """

        def __init__(self):
            """Return built Location Header"""

            self.allItems = "<items>"

        # Since Python Strings are immutable, we need to create a new string. Python does not have an indexed insert
        # or replace, so I create a new string knowing the index where I need to add them
        def addItem(self, name, value, dataType):
            self.allItems = self.allItems + "<item name=\"" + str(name) + "\" value=\"" + str(
                value) + "\" dataType=\"" + str(dataType) + "\" />"

        def addItems(self):
            self.allItems = self.allItems + "</items>"
            MESConnection.bodyItems = self.allItems

    class LocationHeader:
        """ The location of the station:

        Attributes:
            lineNo: Line Number;
            statNo: Station Number;
            statIdx: Station Index, used to differentiate parallel stations. These identical stations should have the
                same Station Numbers and different Station Indexes;
            fuNo: Function Number, currently not being used in TT;
            workPos: Work Position. In TT, for machine events should be set to "0". For other events, such as
                PartReceived or PartProcessed, should be set to "1";
            toolNo: Tool Position. In TT, for machine events should be set to "0". For other events, such as
                PartReceived or PartProcessed, should be set to "1";
            processNo: The number attributed for the process. For processes with a defined standard, the "processNo"
                must use the number previously defined;
            processName: The name attributed for the process. For processes with a defined standard, the "processName"
                must use the standard name previously defined;
            application: The "application" field is also not being used in TT. It should always be assigned the value
                "PLC".
        """

        def __init__(self, lineNo, statNo, statIdx, fuNo, workPos, toolPos, processNo, processName, application="PLC"):
            """Return built Location Header"""

            self.lineNo = lineNo
            self.statNo = statNo
            self.statIdx = statIdx
            self.fuNo = fuNo
            self.workPos = workPos
            self.toolPos = toolPos
            self.processNo = processNo
            self.processName = processName
            self.application = application

    class Header:
        """
        The Header of the Telegram
        If there is a manual timestamp and it is not in a correct format, it will not add it
        Attributes:
            eventSwitch: Used to select the correct processing when multiple processes are possible for the event. On
                the server side, for a certain event, you can have several processes defined and this is the way to
                distinguish them;
            eventName: Name used to identify the event. The Event type can be, for example, "PartProcessed";
            timeStamp: Timestamp of the sent telegram. Not necessary to input because the MES automatically inserts the
                current timestamp but it is useful to send events in the Past.
                Example: "2019-08-30T14:35:32.3234695+01:00";
            version: Version of the message protocol header. You can keep the value as "2.1", it is the latest version;
            eventId: Unique number that identifies the event. Sometimes useful for debugging. MES does not really read
                this camp, you can always set to "1";
            contentType: Defines the format in which the client expects a response. Currently we are using
                "contentType=3" due to the standard body items structures in the telegrams.
        """

        def __init__(self, eventSwitch, eventName="partProcessed", timeStamp="", version="2.1", eventId="1",
                     contentType="3"):
            """Return built Header"""

            self.eventSwitch = eventSwitch
            self.eventName = eventName

            if timeStamp == "":
                self.timeStamp = timeStamp
            else:
                validationResult = self.ValidateTimeStamp(timeStamp)
                if validationResult:
                    self.timeStamp = timeStamp
                else:
                    self.timeStamp = ""

            self.version = version
            self.eventId = eventId
            self.contentType = contentType

        def ValidateTimeStamp(self, timeStamp):
            """ Checks if timestamp is valid. Returns boolean value with result """
            try:
                # Example:"2019-08-30T14:35:32.3234695+01:00"
                datetime.datetime.strptime(timeStamp, '%Y-%m-%dT%X:%f')
                return True
            except ValueError:
                print(
                    "Incorrect data format, should be '%Y-%B-%d%h:%m:%s:%ms' "
                    "Example:\"2019-08-30T14:35:32.3234695+01:00\" ")
                return False

    class ResultHeader:
        """
        This element stores all the event-specific information.
        Currently supports "ResultHeader"

        ResultHeader:
            Attributes:

                result: Contains the result value of the operation.
                        It is the overall result of the process relative to the workpiece
                        Below you can find a table with the corresponding codes:

                    Result 	Description
                    -1 	    No status
                    0 	    Not measured
                    1 	    Good
                    2 	    Bad
                    3 	    Interrupt
                    4 	    Result too small
                    5 	    Result too large
                    6 	    Range too large
                    7 	    Time expired
                    8 	    String comparison incorrect
                    9 	    Measured
                    10 	    Outlier
                    11 	    Information not used
                    12 	    Scrapped
                    255 	Incomplete transmission

                typeNo: A string containing the product number you wish to send to MES.
                typeVar: Type Variant. Distinguishes from different variants of the same part number. By default, may be
                    omitted or assigned an empty string.
                typeVersion: Type Version. Distinguishes from different versions of the same part number. By default,
                    may be omitted or assigned an empty string.

                workingCode: Contains the Processing Code for the type of part to be stored in the MES
                            Below you can find a table with the corresponding codes:

                    WorkingCode 	Description
                    0 		        Series part
                    1 		        Test part
                    2 		        Sample part
                    3 		        Repair part
                    4 		        Calibration part
                    5 		        Master part
                    6 		        Stability part
                    7 		        Changeover part
                    8 		        Data exchange (between stations, programs, etc.)
                    9 		        Empty WPC
                    10 		        Part for CG measurement
                    11 		        Part for SM measurement
                    12 		        Audit WPC
                    13 		        Part for GRR measurement
                    14 		        Warmup WPC
                    15 		        Golden Device

                cycleTimePrev: Cycle time for the previous part in milliseconds.
                nioBits: Bit-coded information if an error occurs. If everything is okay, set to 0
                workCycleCount: Processing counter (rework counter)

         """

        def __init__(self, result, typeNo, typeVar="", typeVersion="", workingCode="0", nioBits="0", workCycleCount="0",
                     cycleTimePrev="0"):
            """Return built Header"""

            self.result = result
            self.typeNo = typeNo
            self.typeVar = typeVar
            self.typeVersion = typeVersion
            self.workingCode = workingCode
            self.nioBits = nioBits
            self.workCycleCount = workCycleCount
            self.cycleTimePrev = cycleTimePrev

    def CreateTelegram(self):
        """
        Constructs the telegram to be sent
        Uses a header, locationHeader and BodyTelegram instances
        Uses the identifier set beforehand. If no identifier was set, it is assigned the value "test"
        """

        def __init__(self):
            self.returnCode = 2

        """ Add Header """
        telegram = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><root><header eventId=\"" + self.header.eventId +\
                   "\" eventName=\"" + self.header.eventName + "\" version=\"" + self.header.version + \
                   "\" eventSwitch=\"" + self.header.eventSwitch + "\" contentType=\"" + self.header.contentType + "\""

        """ If the user decided to input the timestamp, we add it here """
        if self.header.timeStamp == "":
            telegram += ">"
        else:
            telegram += " timeStamp=\"" + self.header.timeStamp + "\">"

        """ Add Location Header """
        telegram += "<location lineNo=\"" + self.locationHeader.lineNo + "\" statNo=\"" + self.locationHeader.statNo + "\" statIdx=\"" + self.locationHeader.statIdx + "\" fuNo=\"" + self.locationHeader.fuNo + "\" workPos=\"" + self.locationHeader.workPos + \
                    "\" toolPos=\"" + self.locationHeader.toolPos + "\" processNo=\"" + self.locationHeader.processNo + \
                    "\" processName=\"" + self.locationHeader.processName + \
                    "\" application=\"" + self.locationHeader.application + "\"/>"

        """ Add event information, including Serial Number """
        telegram += "</header><event><partProcessed identifier=\"" + \
                    self.identifier + "\" /></event>"

        if self.resHeadEnabled:
            """Add body of the message with Result Header"""
            telegram += "<body><structs>" + self.arrayItems + "<resHead result=\"" + self.BodyTelegram.result + "\" typeNo=\"" + self.BodyTelegram.typeNo + "\" typeVar=\"" + self.BodyTelegram.typeVar + "\" typeVersion=\"" + \
                        self.BodyTelegram.typeVersion + "\" workingCode=\"" + self.BodyTelegram.workingCode + "\" nioBits=\"" + \
                        self.BodyTelegram.nioBits + "\" cycleTimePrev=\"" + \
                        self.BodyTelegram.cycleTimePrev + "\"/></structs>"

        try:
            telegram += '<structArrays>' + MESConnection.arrayItems + '</structArrays>'
        except:
            pass

        telegram += "</body></root>"
        return telegram

    def BuildTelegram(self, telegram):
        """Message is processed and transformed into a byte array

            Inputs:
                telegram: The telegram to be sent
        """

        try:
            # Nexeed MES expects the size of the message before the message itself is sent. Add 4 to take into
            # account the initial bytes that contain the size
            size = len(telegram) + 4

            # Bitwise operations to calculate the size of the message in 4 bytes
            lolo = (size & 0xff)
            hilo = ((size >> 8) & 0xff)
            lohi = ((size >> 16) & 0xff)
            hihi = (size >> 24)

            # The message is sent as an array of bytes
            telegramBytes = bytearray()

            # The size of the message is concatenated with the message itself
            telegramBytes.append(hihi)
            telegramBytes.append(lohi)
            telegramBytes.append(hilo)
            telegramBytes.append(lolo)
            telegramBytes += telegram.encode('utf-8')

            return telegramBytes

        except Exception as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            # __str__ allows args to be printed directly, but may be overridden in exception subclasses
            print(inst)
            return -1

    class customArray:
        """
            to be able to create custom array according to needs
            
            usage,
            
            customArray = MESConnection.customArray(arrayName="ExampleName")
            
            arrayName - array name that is wanted to create
             
            rest is same with TestVision class,
            
            name - Screw Confidence
            value - Confidence value for the prediction
            resultState - If 1, operation was successful. If 2,
            unit - In this case, it would be the percentage
            locDetails - additional data
            """

        def __init__(self, arrayName):
            self.arrayName = arrayName

            self.allItems = ("<array name =\"{}\">".format(self.arrayName) +
                             "<structDef >" +
                             "<item name =\"name\" dataType =\"8\" />" +
                             "<item name =\"value\" dataType =\"4\" />" +
                             "<item name =\"unit\" dataType =\"8\" />" +
                             "<item name =\"resultState\" dataType =\"3\" />" +
                             "<item name =\"locDetails\" dataType =\"8\" />" +
                             "</structDef>" +
                             "<values>")

        # Since Python Strings are immutable, we need to create a new string. Python does not have an indexed insert
        # or replace, so I create a new string knowing the index where I need to add them
        def addItem(self, name, value, resultState, unit, locDetails):

            try:
                val = int(value)
                resultState = int(resultState)
            except ValueError:
                raise Exception(
                    "Error, please check if \"value\" and \"resultState\" are int")

            self.allItems += self.allItems + "<item name=\"" + str(name) + "\" value=\"" + str(value) + "\" unit=\"" + \
                            str(unit) + "\" resultState=\"" + str(resultState) + \
                            "\" locDetails=\"" + str(locDetails) + "\" />"

        def addItems(self):
            self.allItems = self.allItems + "</values></array>"
            MESConnection.arrayItems = self.allItems
            return MESConnection.arrayItems  # <- This line was missing

    class ResultTelegram:
        """ The location of the station:

        Attributes:
        Return Code: If "0" everything is fine. Errors are "-1"
        Code: Error number that corresponds to the error;
        Level: Info, Warning or Error;
        Source: Origin of the error;
        Text: "Error Description"
        """

        def ProcessResponse(self, response):
            # Remove the first bytes (they are just the size of the message). Normally they are four but sometimes
            # they are three
            index = response.find('<')
            newdata = response[index:]

            # Enconde again before parsing
            # newdata = newdata.encode()

            # Parse
            root = ET.fromstring(newdata)

            # Get Children of the Root structure
            children = list(root)
            responseHeader = children[0]
            responseEvent = children[1]
            responseBody = children[2]

            # Get Children of the "Event" structure
            responseEventResult = list(responseEvent)

            # Get Return Code. "0" is good, "-1" is bad
            self.returnCode = responseEventResult[0].items()[0][1]

            # If there was an issue, we have to discover the reason
            if self.returnCode == '-1':
                # This is a dict with all the information
                trace = dict(list(responseEventResult)[1][0].items())
                self.code = trace["code"]
                self.level = trace["level"]
                self.source = trace["source"]
                self.text = trace["text"]
                return trace
            else:
                return 0


"""
header = MESConnection.Header("-1")
locationHeader = MESConnection.LocationHeader(
    "8424", "263", "1", "1", "1", "1", "1", "Model", "Raspberry")

BodyTelegram = MESConnection.ResultHeader("1", "test")

# bodyTelegram = MESConnection.BodyTelegram()

mesConnection = MESConnection(
    header, locationHeader, BodyTelegram, "SERIAL")

mesConnection.message = mesConnection.CreateTelegram()

print("Hello")
"""
