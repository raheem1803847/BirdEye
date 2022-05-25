import sqlite3
import os

class Database:
    def open(self):
        self.conn = sqlite3.connect("database.db")
        self.c = self.conn.cursor()
        print(self.conn)

    def close(self):
        self.conn.close()

    def table(self, name):
        "Create a table"
        self.c.execute(f'''CREATE TABLE {name}(
                     date text,
                     account text,
                     debit real,
                     credit real,
                     diff real
        )''')

    def add_row(self, tablename, username , password):
        " Insert a row of data"
        self.c.execute(f"INSERT INTO {tablename} VALUES ({username},{password})")

    def query(self, tablename, column=""):
        print(column)
        if column == "":
            for row in self.c.execute(f'SELECT * FROM {tablename}'):
                print(row)
        else:
            for row in self.c.execute(f'SELECT * FROM {tablename} ORDER BY {column}'):
                print(row)

    def commit(self):
        self.conn.commit()


