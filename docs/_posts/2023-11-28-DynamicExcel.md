---
layout: post 
title: "Reading and Writing to Excel Sheets Dynamically" 
date: 2023-11-18
--- 
For work, I had an excel spreadsheet of data that had about ten input values that were then fed through a TON of nested formulas to get some output value. I needed to do this all without using / opening Microsoft Excel or LibreOffice

A lot of the available libraries in Python can either read in the formulas OR read in the data, but not both operations simultaneously. An application (Excel / LibreOffice) would need to be opened or the file would have to be manually saved. 

This Stack Overflow issue sums up what I was struggling with : [updating and saving excel file...](https://stackoverflow.com/questions/73851931/updating-and-saving-excel-file-using-openpyxl-and-then-reading-it-gives-none-val)


openpyxl is good for interacting with spreadsheets i.e. reading and writing. pycel is good for turning cells into executable python code. 

The solution was to use pycel to turn the spreadsheets into executable code, the using openpyxl to manipulate the cell values. Saving and closing with openpyxl, then turning the sheets back into executable code allows us to see updated outputs after the inputs passed through the formulas. 

One of the comments on this [issue](https://stackoverflow.com/questions/66998366/can-a-pycel-object-be-saved-as-an-excel-workbook) reads :

"...while openpyxl has the computed values available for formula cells for a workbook it read in, it does not really allow those computed values to be saved back into a workbook it writes"

```python 

excel = ExcelCompiler(filename="myFile.xlsx")

# B23 is a cell that contains a value that was calculated by some number of formulas 
originalValue = excel.evaluate("SheetNumber1!B23")

wb = load_workbook('myFile.xlsx',data_only=False)

sheet = wb['SheetNumber1']

# modify your data in whatever way needed 

# say you want to change cell B1 in SheetNumber1 to be 5 
sheet['B1'].value = 5 

# save and close 
wb.save("myFile.xlsx")

wb.close()

 # read in data 
excel = ExcelCompiler(filename="myFile.xlsx")

# this contains the new value after changing cell B1 and running it through formulas 
updatedValue = excel.evaluate("SheetNumber1!B23")

``` 

The time it takes for this code to run is dependent on the amount of data you have in your spreadsheet. The spreadsheet I was working with had a lot of data to be calculated so it took about 45 seconds to load the spreadsheets both times (one to change formulas, the other time to read in the updated data). 

The less data you have to read, the less time it takes to open the file. 

This code works with any system and doesn’t require a download of LibreOffice or Microsoft Excel. 
