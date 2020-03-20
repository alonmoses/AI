# demonstrate template string functions

from string import Template

def main():
    # Usual string formatting with format()
    str1 = "You're watching {0} by {1}".format("Advanced Python", "Joe Marini")
    print(str1)
    
    # TODO: create a template with placeholders
    temp = Template("You're watching ${title} by ${author}")

    # TODO: use the substitute method with keyword arguments
    str2 = temp.substitute(title = "Advanced Python", author = "Alon Moses")
    print(str2)

    # TODO: use the substitute method with a dictionary
    data = {"title" : "Advanced Python", "author" : "Alon Moses"}
    str3 = temp.substitute(data)
    print(str3)
    
if __name__ == "__main__":
    main()
    