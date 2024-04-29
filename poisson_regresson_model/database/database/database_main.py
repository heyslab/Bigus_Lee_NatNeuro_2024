import os.path
import time
import dateutil.parser
import re
import itertools as it

import pkg_resources
import sqlite3
import json
import yaml


class ExperimentDatabase:
    """An object to wrap MySQLdb and handle all connections to mySQL server

    Example
    -------

    >>> from lab.classes import database
    >>> db = database.ExperimentDatabase()
    >>> db.select_all('SELECT * FROM mice LIMIT 10')
    >>> db.disconnect()
    """

    def __init__(self):
        """Connects to SQL database upon initialization."""
        with open(pkg_resources.resource_filename(__name__, '../config.yaml'),
                  'r') as f:
            self._filename = yaml.safe_load(f)[0]['filename']

        self.connect()

    def connect(self):
        """Connect to the SQL database."""
        self._database = sqlite3.connect(self._filename)
        self._database.row_factory = sqlite3.Row

    def disconnect(self):
        """Close the connection to SQL database server."""
        self._database.close()

    def select(self, sql, args=[], verbose=False):
        """ Queries SQL database return single result

        Parameters
        ----------
        sql : str
            Raw SQL to pass to the database.
        args : list, optional
            List of variables to sub into SQL statement in accordiance see
            MySQLdb. Defaults to an empty list.
        verbose : bool, optional
            If set to True print the SQL query. Defaults to False

        Returns
        -------
        result : dict
            First record to match SQL query as a dictionary, with the field
            name being the key.
        """

        cursor = self._database.cursor()
        cursor.execute(sql, args)
        if verbose:
            print(sql)

        result = cursor.fetchone()
        if result is None:
            return None

        return {k:v for (k, v) in zip(result.keys(), result)}


    def select_all(self, sql, args=[], verbose=False):
        """ Queries SQL database and return all the results

        Parameters
        ----------
        sql : str
            Raw SQL to pass to the database.
        args : list, optional
            List of variables to sub into SQL statement in accordiance see
            MySQLdb. Defaults to an empty list.
        verbose : bool, optional
            If set to True print the SQL query. Defaults to False

        Returns
        -------
        result : dict
            All records to match SQL query as a dictionary, with the field name
            being the key.
        """

        cursor = self._database.cursor()
        cursor.execute(sql, args)
        if verbose:
            print(sql)

        result = cursor.fetchall()
        return [{k:v for (k, v) in zip(r.keys(), r)} for r in result]

    def query(self, sql, args=[], verbose=False, ignore_duplicates=True):
        """ Run an arbitrary SQL query i.e. INSERT or DELETE commands. SQL
        statement is passsed directory MySQLdb.

        Parameters
        ----------
        sql : str
            Raw SQL statement to query the database with.
        args : list, optional
            List of variables to sub into SQL statement in accordiance see
            MySQLdb. Defaults to an empty list.
        verbose : bool, optional
            If set to True print the SQL query. Defaults to False
        ignore_duplicates : bool, optional
            If set to True Duplicate entry errors are prevented from being
            raised, instead of raising an error the method returns False.
            Defaults to True.

        Returns
        -------
        bool : True if the SQL statement executes, False otherwise
        """

        cursor = self._database.cursor()
        try:
            cursor.execute(sql, args)
        except Exception as e:
            if not ignore_duplicates or 'Duplicate entry' not in e.__str__():
                raise e
            return False
        else:
            if verbose:
                print(sql)
            self._database.commit()
        return True

    def table_columns(self, table_name):
        res = self.select("SELECT sql FROM sqlite_master WHERE tbl_name = ?",
                          [table_name])

        fields = [a[0] for a in map(
                  re.search, it.repeat('([\w\-]+)'),
                  re.sub('\s+', ' ',
                         re.search('(?<=\()([^(^)]+)', res['sql'])[0]).split(','))]
        return fields


def reset_db():
    """ Deletes all records an resets the database schema """

    db = ExperimentDatabase()
    db.disconnect()
    os.remove(db._filename)
    db.connect()

    db.query("""
        CREATE TABLE trials(
            trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INT,
            experiment_type TEXT,
            mouse_id INT,
            data_folder TEXT,
            start_time INT
            )""")

    db.query("""
        CREATE TABLE trial_attributes(
            attribute_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id int,
            attribute text,
            value text
            )""")

    db.query("""
        CREATE TABLE projects(
            project_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name text
            )""")

    db.query("""
        CREATE TABLE project_attributes(
            attribute_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id int,
            attribute text,
            value text
            )""")

    db.query("""
        CREATE TABLE mice(
            mouse_id INTEGER PRIMARY KEY AUTOINCREMENT,
            mouse_name text
            )""")

    db.query("""
        CREATE TABLE mouse_attributes(
            attribute_id INTEGER PRIMARY KEY AUTOINCREMENT,
            mouse_id int,
            attribute text,
            value text
            )""")

    db.disconnect()
