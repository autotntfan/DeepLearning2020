import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
import time
import predealimg
import h5py as h5
import scipy.io
import os
#這是用來前處理圖片的多進程處理，使用前請將處理好的資料.h5檔刪除才會動作!
if __name__=='__main__':
    datatype=str(input("test or train?"))
    if not os.path.exists('DLmidterm/'+datatype+'.h5'): 
        t_start=time.time()
        data=scipy.io.loadmat('DLmidterm/'+datatype+".mat")
        z=np.float32(np.zeros((data["x"].shape[1],48,140,3)))
        tasks=[(i,datatype)for i in list(range(int(data["x"].shape[1])))]
        p=mp.Pool()
        res=[p.starmap(predealimg.Deal_With_Img,tasks)]
        p.close()
        p.join()
        z=(np.asarray(res[0]))
        with h5.File('DLmidterm/'+datatype+'.h5','w') as file:
            file.create_dataset('x',data=z)
        t_End=time.time()
        print(t_End-t_start)
    
    with h5.File('DLmidterm/'+datatype+'.h5','r') as file:
        x=file.get('x').value
    plt.imshow(x[4])
    plt.show() #確認圖片是否正確處理完成

    