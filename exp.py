import multiprocessing
import threading
import time

queue = multiprocessing.Manager().Queue(maxsize=1000)
l = []

def producer(index):
    time.sleep(1)
    queue.put(index)

def consumer():
    while True:
        l.append(queue.get())

def main():
    pool = multiprocessing.Pool(10)
    threading.Thread(target=consumer).start()
    for i in range(20):
        pool.apply_async(producer, args=(i,))
    
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
    print(l)
