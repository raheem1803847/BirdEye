import sqlite3
import os

##create database called database
conn = sqlite3.connect("database.db")

# create a cursor
c = conn.cursor()

# create table , between """ """ write command
#datatypes in sqllite :NULL - INTEGER - REAL(dicemal) - TEXT - BLOB(image-video)

#c.execute("""CREATE TABLE accounts (
#        user_name text,
#        password text ,
#        user_type text
#
#        )""")
#

#insert one record
#c.execute("INSERT INTO accounts VALUES( 'Alaa','12345678','student')")
#------------------------------------------------
#c.execute("SELECT * FROM accounts")
#print(c.fetchall())
##print(c.fetchone()) #fetch the last one



#% means start with Alaa then end by any thing
#c.execute("DELETE FROM accounts")
c.execute("SELECT * FROM accounts ")


print(c.fetchall())


##to commit the excute by (push to database)
conn.commit()

#then close the connection
conn.close()
