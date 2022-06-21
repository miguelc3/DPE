import threading
import os
import subprocess


def task1():
    cmd = 'python AcquireAndDisplay.py'
    p = subprocess.Popen(cmd, shell=True)
    out, err = p.communicate()


def task2():
    cmd = 'python auto_system.py'
    p = subprocess.Popen(cmd, shell=True)
    out, err = p.communicate()


def main():
    # print ID of current process
    print("ID of process running main program: {}".format(os.getpid()))

    # print name of main thread
    print("Main thread name: {}".format(threading.current_thread().name))

    # creating threads
    t1 = threading.Thread(target=task1, name='t1')
    t2 = threading.Thread(target=task2, name='t2')

    # starting threads
    t1.start()
    t2.start()

    # wait until all threads finish
    t1.join()
    t2.join()


if __name__ == "__main__":
    main()
