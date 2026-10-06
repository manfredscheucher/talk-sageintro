import time
from datetime import datetime        
from multiprocessing import Pool

def now(): 
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def f(i):
    time.sleep(float(0.1))
    return i

if __name__ == '__main__':
    data = list(range(50))
    print(now(),"single-thread start")
    print(list(map(f, data)))
    print(now(),"single-thread stop")

    with Pool() as pool:
        print(now(),"parallel start")
        print(list(pool.map(f, data)))
        print(now(),"parallel stop")
        

