# Demonstrate how to customize logging output

import logging

userExt = {
    "user" : "Alon Moses"
}

# TODO: add another function to log from
def oneMoreFunc():
    logging.debug("a debug message", extra = userExt)


def main():
    # set the output file and debug level, and
    # TODO: use a custom formatting specification
    fmtStr = "Time:%(asctime)s: %(levelname)s:Function:%(funcName)s" \
             " Line:%(lineno)d %(message)s User: %(user)s"
    dateStr = "%m/%d/%Y %I:%M:%S %p"
    logging.basicConfig(filename="output.log",
                        level=logging.DEBUG,
                        filemode="w",
                        format=fmtStr,
                        datefmt=dateStr)

    logging.info("This is an info-level log message", extra = userExt)
    logging.warning("This is a warning-level message", extra = userExt)
    oneMoreFunc()

if __name__ == "__main__":
    main()
