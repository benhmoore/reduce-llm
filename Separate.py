# Build array of lines from file, strip newlines
import re

filename = input("Enter a filename: ")

str = "computer"
pat = re.compile(r"\b\b", re.IGNORECASE)  # upper and lowercase will match

mylines = []                                # Declare an empty list.
with open (filename, 'rt', encoding='utf-8') as myfile:    # Open lorem.txt for reading text.
    for myline in myfile:                   # For each line in the file,
        if myline.find(str) != -1:
            mylines.append(myline.rstrip('\n'))
        #if pat.search(str) != None:
            #print("Found it.")
        #    mylines.append(myline.rstrip('\n')) # strip newline and add to list.
print(mylines)





