import sys

import database


def main(args):
    database.reset_db()
    db = database.ExperimentDatabase()

    #folder = '/Users/jbowler/Lab2/data/tDNMS_task_behavior/4.4.22.A1.task_/'
    #database.update_trial(folder, 'A1', 'tDNMS', 'tDNMS', '4/4/22')

if __name__ == '__main__':
    main(sys.argv[1:])
