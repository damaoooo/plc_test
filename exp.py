import multiprocessing
import random
import time



def crash(index):
    s = random.randint(1, 3)
    time.sleep(1)
    print(index, "is runninng")
    if s == 1:
        raise ValueError
    
    elif s == 2:
        raise IndexError
    
    else:
        raise KeyError

def start_up(pool):
    for i in range(100):
        try:
            print("Time after apply is", time.time())
            s = pool.apply_async(func=crash, args=(i, ))
            
        except ValueError as e:
            print("ValueError at", i)
            continue
        
        except KeyboardInterrupt as e:
            print("KeyboardInterrupt at", i)
            exit(1)

        except:
            print("Exception at", i)
            continue
    pool.close()
    pool.join()

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=10)
    start_up(pool)
